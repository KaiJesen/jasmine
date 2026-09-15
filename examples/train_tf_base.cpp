#include <iostream>
#include "jas_transformer_t.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_loss_t.hpp"


using namespace jasmine;
int main()
{
    using val_type = double;
    using net_type = complex_net_builder_t<val_type>
        ::template push_back_updatable<transformer_base_t, base_upr_tpl>
        ::template push_back_updatable<weight_net_t, base_upr_tpl>
        ::template push_back_staticnet<sigmoid_net_t>
        ::template push_back_staticnet<mse_loss_t>
        ::type;

    net_type cnet;
    auto& tf_base = cnet.get<0>();
    tf_base.set_param(2, 3, 2, 4, 16);
    auto& ffn = cnet.get<1>();
    ffn.reinit({4, 4});

    mat_t<val_type> en_input(4, 2, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
    mat_t<val_type> de_input(4, 2, {0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2});
    mat_t<val_type> label(4, 2, {
        0.3, 0.2,
        0.1, 0.8,
        0.5, 0.1,
        0.7, 0.8
    });

    double lr = 0.01;
    std::cout << "Input learning rate: ";
    std::cin >> lr;
    int train_steps = 200;
    std::cout << "Input train steps: ";
    std::cin >> train_steps;

    cnet.set_updator(lr);
    cnet.init_weight<xavier_gaussian_t>();
    tf_base.encoder_forward(en_input);

    auto output = cnet.forward(de_input);
    std::cout << "Before training Output:\n" << output << std::endl;
    for (int i = 0; i < train_steps; ++i)
    {
        output = cnet.forward(de_input);
        cnet.backward(label);
        cnet.step();
    }
    std::cout << "After training Output:\n" << output << std::endl;
    return 0;
}
