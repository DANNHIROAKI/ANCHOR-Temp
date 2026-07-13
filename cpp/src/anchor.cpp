#include "anchor/algorithms.hpp"

template class anchor::BoxSet<double>;
template class anchor::BoxSet<std::int64_t>;
template class anchor::AnchorCompiled<double>;
template class anchor::AnchorCompiled<std::int64_t>;
template class anchor::AnchorStreaming<double>;
template class anchor::AnchorStreaming<std::int64_t>;
template class anchor::LiftedRangeTree<double>;
template class anchor::LiftedRangeTree<std::int64_t>;
template class anchor::SweepRangeTree<double>;
template class anchor::SweepRangeTree<std::int64_t>;
