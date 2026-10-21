from environment.field import generate_field
from quantum.amplitude_encoding import verify_encoding


# --------------------------------------------------
# Same environment as classical baseline
# --------------------------------------------------

agents = [
    (2.0, 2.0),
    (3.0, 13.0)
]

goals = [
    (13.0, 13.0),
    (13.0, 2.0)
]

obstacles = [
    (8.0, 9.0),
    (5.0, 11.0)
]


# --------------------------------------------------
# Generate field for Robot 0
# --------------------------------------------------

X, Y, field = generate_field(
    agents=agents,
    goals=goals,
    obstacles=obstacles,
    current_agent=0
)


# --------------------------------------------------
# Encode and verify
# --------------------------------------------------

circuit, statevector = verify_encoding(
    field
)

print("\nQuantum circuit:")
print(circuit)