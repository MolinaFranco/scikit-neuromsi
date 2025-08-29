# Crear modelo
model = Echeveste2020(N_E=50,
N_I=50)

# Cargar parámetros 
# optimizados de Echeveste
connectivity_params = {
    'a_EE': 0.331089, 'a_EI':
0.081307, 'a_IE': 1.234567,
'a_II': 0.567890,
    'd_EE': 0.802756, 'd_EI':
0.543210, 'd_IE': 0.987654,
'd_II': 0.456789
}

# Construir matriz W una sola vez
W = model.build_connectivity_matrix(connectivity_params)
# Usar W en simulaciones (esto
#  iría dentro de model.run())
response = model._integrator(u_e, u_i, t, W, h, eta)