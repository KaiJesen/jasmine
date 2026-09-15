#include <iostream>
#include "examples/transformer_mse_demo.hpp"

using namespace jasmine;

int main()
{
    mse_transformer_demo_t net;
    net.init(1e-5);
    int input_dim = mse_transformer_demo_t::input_dim;

    mat_t<test_val_type> en_input(input_dim, 3, {
        0.5, 0.8, 0.3,
        0.7, 0.2, 0.4,
        0.6, 0.8, 0.1,
        0.9, 0.3, 0.7,
        0.2, 0.6, 0.4,
        0.1, 0.9, 0.5,
        0.3, 0.5, 0.8,
        0.8, 0.1, 0.6
    });
    mat_t<test_val_type> de_input(input_dim, 3, {
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9,
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.9, 0.1, 0.8,
        0.2, 0.3, 0.5,
        0.6, 0.7, 0.4,
        0.5, 0.9, 0.2
    });
    mat_t<test_val_type> label(input_dim, 3, {
        0.3, 0.2, 0.1,
        0.8, 0.5, 0.1,
        0.7, 0.8, 0.9,
        0.4, 0.3, 0.2,
        0.6, 0.2, 0.5,
        0.9, 0.1, 0.3,
        0.5, 0.7, 0.4,
        0.1, 0.6, 0.8
    });

    int train_times = 100000;
    std::cout << "MSE demo (continuous + SOS/EOS flags)\n";
    std::cout << "Input train times: ";
    std::cin >> train_times;

    net.train(en_input, de_input, label, train_times);
    net.predict(en_input);
    return 0;
}
