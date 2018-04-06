package fr.univlyon1.memory.prioritizedExperienceReplay;

import java.util.Comparator;

public class InteractionComparatorProb<E> implements Comparator<E> {
    StochasticPrioritizedExperienceReplay spe ;

    public InteractionComparatorProb(StochasticPrioritizedExperienceReplay spe){
        super();
        this.spe = spe ;
    }

    @Override
    public int compare(E e, E t1) {
        Double val = this.spe.getSum();
        InteractionHistory ih1 = (InteractionHistory) e;
        InteractionHistory ih2 = (InteractionHistory) t1;
        Double val1 = ih1.getErrorValue() / val ;
        Double val2 = ih2.getErrorValue() / val ;
        if (val1 < val2)
            return -1;
        if (val1.equals(val2))
            return 0;
        return 1;
    }
}