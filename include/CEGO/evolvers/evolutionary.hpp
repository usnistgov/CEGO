///// Do a N-element tourney selection to generate a given length of individuals
//Population tourney_selection(Population &this_layer, Population &older_layer, std::size_t length, std::size_t Ntourney)
//{

//    // see http://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
//    std::uniform_real_distribution<> float_dis(0, 1);
//    std::uniform_int_distribution<> this_int_selector = (this_layer.size() == 0) ? std::uniform_int_distribution<>(0, 1) : std::uniform_int_distribution<>(0, static_cast<int>(this_layer.size()) - 1);
//    std::uniform_int_distribution<> older_int_selector = (older_layer.size() == 0) ? std::uniform_int_distribution<>(0, 1) : std::uniform_int_distribution<>(0, static_cast<int>(older_layer.size()) - 1);

//    auto &gen = get_gen();

//    Population out;

//    for (auto counter = 0; counter < length; ++counter) {
//        // get the list of candidates for the tournament selection;
//        std::vector<pIndividual > candidates;
//        std::vector<double> objectives;
//        for (auto j = 0; j < Ntourney; ++j)
//        {
//            if (older_layer.size() == 0 || float_dis(gen) < 0.8) {
//                // Pull from this generation
//                candidates.push_back(this_layer[this_int_selector(gen)]->copy());
//            }
//            else {
//                // Pull from the old generation
//                candidates.push_back(older_layer[older_int_selector(gen)]->copy());
//            }
//            objectives.push_back(candidates.back()->get_cost());
//        }
//        // Find the individual with the lowest objective function (actually, an iterator to it)
//        auto min_el = std::min_element(objectives.begin(), objectives.end());

//        // Get the index of the winner of the tourney
//        std::size_t j = min_el - objectives.begin();

//        // Get the index of the other one from the tourney (might be the same)
//        std::size_t j2 = std::uniform_int_distribution<std::size_t>(0, Ntourney - 1)(gen);

//        // Add it to the outputs
//        out.emplace_back(candidates[j]->recombine_with(candidates[j2], m_bounds, RecombinationFlags()));
//    }
//    return out;
//};