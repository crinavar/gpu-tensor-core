Los fuentes están en la carpeta CUB/cub/reduce/codes

--------
Como compilar:

Para compilar, hay que hacerlo desde la ruta CUB/cub/reduce/codes

nvcc -arch=sm_XX source_code.cu -o program_name -I../.. -lcudart -O3


sm_XX = versión arquitectura.


-----------
Como ejecutar:

	./nombre_ejecutable CANTIDAD_NUMEROS_ALEATORIOS


Nota: Todos los codigos realizan 100 iteraciones y luego entregan tiempos promedio (con ElapsedMillis).
	para modificar la cantidad de iteraciones ir a la variable global:

		int g_timing_iterations = 100;

