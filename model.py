import tensorflow as tf
import numpy as np
from qdataloader import QCNDataloader

# ---------------------------------------------
# Basic 1D Residual Block for Capsule Branches
# ---------------------------------------------
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = tf.keras.layers.Conv1D(filters, kernel_size, strides=stride, padding='same', activation='relu',
                                 kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, strides=1, padding='same',
                               kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Adjust shortcut if dimensions change.
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, strides=stride, padding='same',
                                          kernel_initializer='he_uniform')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

# ---------------------------------------------
# Custom EM Routing Layer
# ---------------------------------------------
class EMRoutingLayer(tf.keras.layers.Layer):
    def __init__(self, num_iterations=2, **kwargs):
        super(EMRoutingLayer, self).__init__(**kwargs)
        self.num_iterations = num_iterations

    def call(self, inputs):
        # inputs shape: (batch, num_caps, capsule_dim)
        # We assume capsule_dim = 5 where first 4 dims are pose and last is activation.
        batch_size = tf.shape(inputs)[0]
        num_caps = tf.shape(inputs)[1]
        capsule_dim = tf.shape(inputs)[2]
        
        # Separate pose and activation
        poses = inputs[..., :4]       # (batch, num_caps, 4)
        activations = inputs[..., 4:] # (batch, num_caps, 1)
        
        # For a single output capsule, initialize routing weights uniformly.
        r = tf.fill([batch_size, num_caps, 1], 1.0 / tf.cast(num_caps, tf.float32))
        
        # Iterative EM routing (a simplified version)
        for it in range(self.num_iterations):
            # M-step: Compute weighted mean (capsule pose)
            weighted_sum = tf.reduce_sum(r * activations * poses, axis=1)  # (batch, 4)
            sum_weights = tf.reduce_sum(r * activations, axis=1, keepdims=True)  # (batch, 1)
            capsule_pose = weighted_sum / (sum_weights + 1e-8)  # (batch, 4)
            
            # E-step: Update routing probabilities based on similarity (negative squared distance)
            # Expand capsule_pose for broadcast:
            capsule_pose_exp = tf.expand_dims(capsule_pose, axis=1)  # (batch, 1, 4)
            diff = poses - capsule_pose_exp  # (batch, num_caps, 4)
            dist = tf.reduce_sum(tf.square(diff), axis=-1, keepdims=True)  # (batch, num_caps, 1)
            # Update r: softmax over num_caps dimension (for each batch element)
            r = tf.nn.softmax(-dist, axis=1)
            
        # After routing, also compute an output activation (here we simply use the sum of weights)
        output_activation = tf.squeeze(sum_weights, axis=-1)  # (batch, 1)
        # Concatenate capsule pose and activation to form the output capsule (dimension = 5)
        output_capsule = tf.concat([capsule_pose, output_activation], axis=-1)  # (batch, 5)
        return output_capsule

    def compute_output_shape(self, input_shape):
        # Output shape: (batch, 5)
        return (input_shape[0], 5)

# ---------------------------------------------
# V1 Quaternion Capsule Network Model
# ---------------------------------------------
def build_qcn_v1_model(input_shape=(31, 4)):
    """
    Build the V1 Quaternion Capsule Network model.
    
    Input: A tensor of shape (31, 4) representing the 2D keypoint quaternions.
    Two branches:
      - Pose Branch (deeper) extracts pose information.
      - Activation Branch (shallower) extracts activation probabilities.
    Their outputs are merged to form primary capsules (one per joint, dimension 5).
    An EM routing layer aggregates the 31 primary capsules into one output capsule.
    A Dense layer maps this routed capsule to a 7D vector:
      - First 4: predicted view rotation quaternion (normalized).
      - Last 3: predicted view translation.
    """
    inputs = tf.keras.Input(shape=input_shape)  # (31, 4)

    # ---------- Pose Branch ----------
    # Process the input (treated as a sequence of 31 joints)
    pose = residual_block(inputs, filters=32, kernel_size=3, stride=1)  # output shape: (31, 32)
    pose = residual_block(pose, filters=64, kernel_size=3, stride=2)      # output shape: (ceil(31/2), 64)
    # 1x1 convolution to increase channels to match Primary capsule dimensionality.
    pose = tf.keras.layers.Conv1D(96, kernel_size=1, activation='relu', padding='same',
                                  kernel_initializer='glorot_uniform')(pose)
    pose = tf.keras.layers.BatchNormalization()(pose)
    # To maintain one capsule per joint, we use global average pooling along the temporal (joint) axis.
    # Here, however, we need 31 capsules. We assume that upsampling or interpolation is applied
    # so that the pose branch produces an output of shape (31, 4).
    # For simplicity in this V1 model, we add a Dense layer that maps the pooled features back to 31 capsules.
    pose_flat = tf.keras.layers.GlobalAveragePooling1D()(pose)  # (batch, 96)
    # Map to 31*4 outputs and reshape to (31, 4)
    pose_out = tf.keras.layers.Dense(31 * 4, activation='linear')(pose_flat)
    primary_pose = tf.keras.layers.Reshape((31, 4))(pose_out)  # Each row is a pure quaternion (but may not be unit)

    # ---------- Activation Branch ----------
    # Use a shallower branch
    activation = residual_block(inputs, filters=32, kernel_size=3, stride=2)  # (approx 31/2, 32)
    activation = tf.keras.layers.Conv1D(31, kernel_size=1, activation='sigmoid', padding='same',
                                          kernel_initializer='glorot_uniform')(activation)
    activation = tf.keras.layers.GlobalAveragePooling1D()(activation)  # (batch, 31)
    primary_activation = tf.keras.layers.Reshape((31, 1))(activation)  # (batch, 31, 1)

    # ---------- Form Primary Capsules ----------
    # Concatenate pose and activation along last dimension to get capsules of dimension 5.
    primary_capsules = tf.keras.layers.Concatenate(axis=-1)([primary_pose, primary_activation])  # (batch, 31, 5)

    # ---------- EM Routing ----------
    # Apply the custom EM routing layer to aggregate the 31 capsules into one capsule.
    routed_capsule = EMRoutingLayer(num_iterations=2)(primary_capsules)  # (batch, 5)

    # ---------- Final Output Layer ----------
    # Map the routed capsule to a 7D vector: first 4 for view rotation, last 3 for view translation.
    view_transform = tf.keras.layers.Dense(7, activation='linear', kernel_initializer='glorot_uniform')(routed_capsule)
    view_rot_pred = view_transform[..., :4]  # (batch, 4)
    view_trans_pred = view_transform[..., 4:]  # (batch, 3)

    # Normalize the quaternion part so that it is a unit quaternion.
    view_rot_pred = tf.keras.layers.Lambda(lambda x: x / (tf.norm(x, axis=-1, keepdims=True) + 1e-8))(view_rot_pred)

    model = tf.keras.Model(inputs=inputs, outputs=[view_rot_pred, view_trans_pred])
    return model

# ---------------------------------------------
# Example usage
# ---------------------------------------------
if __name__ == "__main__":
    data_path  = './data/train_data_20.json'
    val_data   = './data/val_data_20.json'
    
    # (Optionally) define the projection matrix if needed.
    projection_matrix = np.array([2.790552,  0.,         0.,         0.,
                                  0.,         1.5696855,  0.,         0.,
                                  0.,         0.,        -1.0001999, -1.,
                                  0.,         0.,        -0.20002,    0.], dtype=np.float32)
    projection_matrix = projection_matrix.reshape((4, 4))
    
    qc_loader = QCNDataloader(data_path, val_data, projection_matrix=projection_matrix)
    sample_qcn = qc_loader.__getitem__(1)
    batch_size = 32 
    train_loader = qc_loader.prepare_data(batch_size,is_training=True)
    for i in range(2):
        batch = next(iter(train_loader))
    model = build_qcn_v1_model()
    model.summary()
    
    dummy_input = batch[0]
    rot_out, trans_out = model.predict(dummy_input)
    print("Predicted view rotation (quaternion):", rot_out.shape)  # Expect (32, 4)
    print("Predicted view translation:", trans_out.shape)          # Expect (32, 3)
