import torch

# Datos iniciales
PrecioGasolinaPorLitro = 20 # Precio por litro de gasolina
CapacidadTanqueL = 40  # Capacidad del tanque en litros

# Función de costo (costo total)
def CostoTotal(CantidadGasolinaLitros):
    """
    Calcula el costo total de llenar el tanque con una cantidad dada de gasolina.
    """
    return CantidadGasolinaLitros * PrecioGasolinaPorLitro 


# Descenso de gradiente
def gradient_descent():
    learning_rate =0.090  # Tasa de aprendizaje
    num_iterations = 5  # Número de iteraciones

    CantidadGasolinaLitros = torch.tensor(CapacidadTanqueL / 2.0, requires_grad=True)  # Conjetura inicial

    for _ in range(num_iterations):
        costo = CostoTotal(CantidadGasolinaLitros)
        costo.backward()  # Calcula el gradiente
        CantidadGasolinaLitros.data -= learning_rate * CantidadGasolinaLitros.grad  # Actualiza los parámetros
        CantidadGasolinaLitros.grad.zero_()  # Reinicia el gradiente

    return CantidadGasolinaLitros.item()

# Resultado
CantidadOptimaGasolina = gradient_descent()
costo_optimo = CostoTotal(CantidadOptimaGasolina)

print(f"La cantidad optima de gasolina es aproximadamente {CantidadOptimaGasolina:.2f} litros.")
print(f"El costo total optimo es de ${costo_optimo:.2f}.")
