#include <pybind11/pybind11.h>

namespace py = pybind11;

class Pet {
    public:
        Pet(std::string name, std::string breed, uint age);
        void pet();
        uint happiness();

    private:
        uint _age;
        std::string _name;
        std::string _breed;
        uint _happiness;
};

Pet::Pet(std::string name, std::string breed, uint age) {
    this->_name = name;
    this->_breed = breed;
    this->_age = age;
    this->_happiness = 0;
}

void Pet::pet() {
    this->_happiness++;
    return;
}

uint Pet::happiness() {
    return this->_happiness;
}


PYBIND11_MODULE(mnpkg, m){
    m.doc() = "Metodos Numericos Package - Implementacion de KNN y PCA";

    py::class_<Pet>(m, "Pet")
        .def(py::init<std::string, std::string, uint>(), py::arg("name"), py::arg("breed"), py::arg("age"))
        .def("happiness", &Pet::happiness)
        .def("pet", &Pet::pet);        

    // py::class_<KNNClassifier>(m, "KNNClassifier")
    //     .def(py::init<uint &>())
    //     .def("fit", &KNNClassifier::fit);
    //     // .def("predict", &KNNClassifier::predict)
    //     // .def("set_params", &KNNClassifier::set_params)
    //     // .def("get_params", &KNNClassifier::get_params);
}
