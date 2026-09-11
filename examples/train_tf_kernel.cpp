#include <iostream>
#include "mat_transformer_kernel_t.hpp"
#include "mat_init_t.hpp"
#include "mat_loss_t.hpp"
#include "mat_updator_t.hpp"


using namespace jasmine;
template <typename val_type>
using upr_tpl = cache_updator_t<val_type, nadam_t>;

int main()
{
    using val_type = double;
    using tf_type = transformer_kernel_t<val_type, upr_tpl>;
    using net_type = complex_net_builder_t<val_type>
        ::push_back_impl<tf_type>
        ::push_back_staticnet<mse_loss_t>
        ::type;

    net_type cnet;
    auto& tf = cnet.template get<0>();
    tf.set_param(3, 2, 2, 4, 16);

    std::cout << cnet.net_type() << std::endl;
    std::cout << "input lr:";
    double lr = 0.01;
    cnet.init_weight<xavier_gaussian_t>();
    std::cin >> lr;
    cnet.set_updator(lr);

    mat_t<val_type> encoder_input(4, 1, {0.1, 0.2, 0.3, 0.4});
    mat_t<val_type> decoder_input(4, 1, {0.1, 0.2, 0.3, 0.4});
    mat_t<val_type> decoder_target(4, 1, {0.1, 0.2, 0.3, 0.4});

    tf.encoder_forward(encoder_input);
    auto output = tf.forward(decoder_input);
    std::cout << "Before training: " << output.to_string() << std::endl;
    for (int i = 0; i < 150; ++i)
    {
        output = cnet.forward(decoder_input);
        cnet.backward(decoder_target);
        cnet.step();
    }
    std::cout << "After training: " << output.to_string() << std::endl;
    return 0;
}
