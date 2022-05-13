// Parser de imagenes. Que reciba el vector con todos los pixeles
// y convierta eso a una Matriz de Eigen
// Ver: https://stackoverflow.com/questions/50883703/pybind-how-can-i-operate-over-a-pylist-object

// TL;DR: usar el tipo `py::list` para comprender una lista. Iterador: `py::handle obj : l`