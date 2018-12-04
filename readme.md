# Reconocimiento de números escritos a mano utilizando el algoritmo ALR3
Es un proyecto desarrollado en el marco de la materia Gestión Avanzada de Datos de la carrera Ingeniería en Sistemas de Información de la Facultad Regional Concepción del Uruguay de la Universidad Tecnológica Nacional de Argentina.

## Objetivo
El objetivo del proyecto es la detección de números escritos a mano a través de la caracterización y comparación de imágenes.

## Estado Actual
Utilizando el algoritmo ALR3 con ciertas modificaciones para intentar distinguir imágenes espejadas (números 5 y 2) o con excesiva rotación (en especial entre los números 6 y 9).
Las modificaciones mejoraron la eficacia del algoritmo pero aun sigue siendo un problema a resolver.

Se realizaron pruebas tomando como consulta el dataset MINIST y se logró una eficacia de poco más del 50 % lo que se puede ser por varias causas:
- El dataset MINIST contiene números dibujados muy distintos a los de la base de datos.
- Falta de ejemplos en la base de datos de los distintos números y sus variantes (actualmente hay menos de 300).

## Contenido del proyecto
- Lógica de pre-procesamiento, caracterización y comparación de imágenes.
- Sitio web para escribir consultas y para agregar elementos a la base de datos.

## Tecnologías utilizadas
- Python
- CUDA
- Javascript

## Artículos relacionados
[Zernike Moments vs ALR3 Applied to Similarity Searching of Cattle Brands](https://www.researchgate.net/publication/328049660_Zernike_Moments_vs_ALR3_Applied_to_Similarity_Searching_of_Cattle_Brands>)

## Contribuciones
- Lautaro Zapata ([Github](https://github.com/lautaro08))