# Number of filters in the first layer of G and D
gf = 32
df = 64
learning_rate = 0.0002

# Loss weights
lambda_cycle = 10.0  # Cycle-consistency loss
lambda_id = 0.1 * lambda_cycle  # Identity loss

ancho = 128
alto = 128
canales = 3
