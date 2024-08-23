# Redes Neuronales para Predicción de Pasajeros de Bus (LSTM)
El propósito de esta tarea es implementar el modelo LSTM para predecir la cantidad de pasajeros de un bus.
Para esto se pide:
1.	De la base de datos otorgada en clases filtrar:
  a.	Para TRAIN los años 2016-2018 
  b.	Para TEST los años 2019 
2.	Normalizar los datos filtrados entre -1 y 1
3.	Experimentar: Con un Batch=32
    Experimentos	RMSE
    Obtener el valor del “loss” con 1 época	
    Obtener el valor del “loss” con 100 épocas	
    Obtener el valor del “loss” con 200 épocas	
    Obtener el valor del “loss” con 400 épocas	
    Obtener el valor del “loss” con 600 épocas	
    Obtener el valor del “loss” con 800 épocas	
    Obtener el valor del “loss” con 1000 épocas	
De esta experimentación obtener:
  a)	Una gráfica de loss(eje y) vs Épocas (eje x)
  b)	El RMSE para cada caso
  c)	Los gráficos resultantes de la serie de tiempo

4.	Del mejor MSE del resultado anterior, experimentar con el mismo nro de épocas los siguientes Batch y obtener el RMSE:
    Experimentos	RMSE
    Obtener el valor del “loss” con 1 Batch	
    Obtener el valor del “loss” con 16 Batch	
    Obtener el valor del “loss” con 32 Batch	
    Obtener el valor del “loss” con 64 Batch	
    Obtener el valor del “loss” con 128 Batch	
    Obtener el valor del “loss” con 512 Batch	
De esta experimentación obtener:
    a)	Una gráfica de loss(eje y) vs Batch (eje x)
    b)	El RMSE para cada caso
    c)	Los gráficos resultantes de la serie de tiempo
5.	Del mejor MSE del resultado anterior, experimentar con el mismo nro de épocas y el mismo número de batch las neuronas de capa oculta del sistema:
    Experimentos	RMSE
    Obtener el valor del “loss” con 10 neuronas capa oculta	
    Obtener el valor del “loss” con 20 neuronas capa oculta	
    Obtener el valor del “loss” con 30 neuronas capa oculta	
    Obtener el valor del “loss” con 40 neuronas capa oculta	
    Obtener el valor del “loss” con 50 neuronas capa oculta	
    Obtener el valor del “loss” con 60 neuronas capa oculta	
De esta experimentación obtener:
    d)	Una gráfica de loss(eje y) vs Neuronas (eje x)
    e)	El RMSE para cada caso
    f)	Los gráficos resultantes de la serie de tiempo

6.	Del mejor MSE del resultado anterior, experimentar con las métricas de optimización 
    Experimentos	RMSE
    ‘sgd’	
    ‘adam’	
    ‘otro’
De esta experimentación obtener:
    g)	El RMSE para cada caso
    h)	el mejor RMSE 
    i)	Los gráficos resultantes de la serie de tiem 
