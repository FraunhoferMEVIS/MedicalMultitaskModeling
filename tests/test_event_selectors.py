import numpy as np
from mmm.event_selectors import (
    EventSelectorBase,
    FixedEventSelector,
    RecurringEventSelector,
    CodedEventSelector,
    CombinedEventSelector,
)


def test_fixed_event_selector():
    selector = FixedEventSelector(at_iterations=[1, 5, 10])
    trues = [selector.is_event(i) for i in range(100)]
    assert trues.count(True) == 3 and trues[5] and not trues[4]


def test_combined_event_selectors():
    s1 = FixedEventSelector(at_iterations=[0, 8, 14])
    s2 = RecurringEventSelector(every_n=3)
    selector = CombinedEventSelector(events=[s1, s2])

    c1s = [s1.is_event(i) for i in range(100)]
    c2s = [s2.is_event(i) for i in range(100)]
    combs = [selector.is_event(i) for i in range(100)]
    assert np.array_equal(np.array(combs), np.array(c1s) | np.array(c2s))
