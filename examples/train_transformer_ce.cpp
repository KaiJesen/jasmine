#include <iostream>
#include "examples/transformer_ce_demo.hpp"

using namespace jasmine;

int main()
{
    ce_transformer_demo_t net;
    net.init(1e-3);

    // copy 任务：内容 token 3,4,5（PAD=0 SOS=1 EOS=2）
    mat_t<double> src(1, 3, {3.0, 4.0, 5.0});
    mat_t<double> label(1, 3, {3.0, 4.0, 5.0});

    int train_times = 2000;
    std::cout << "CE demo (discrete tokens + embedding + CE loss)\n";
    std::cout << "vocab=" << ce_transformer_demo_t::vocab
              << " d_model=" << ce_transformer_demo_t::d_model
              << " PAD/SOS/EOS=" << ce_transformer_demo_t::PAD << "/"
              << ce_transformer_demo_t::SOS << "/"
              << ce_transformer_demo_t::EOS << "\n";
    std::cout << "Input train times: ";
    std::cin >> train_times;

    net.train(src, label, train_times);
    net.predict(src);
    return 0;
}
