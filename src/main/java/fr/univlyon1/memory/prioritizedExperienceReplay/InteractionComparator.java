package fr.univlyon1.memory.prioritizedExperienceReplay;

import java.util.Comparator;

public class InteractionComparator<E> implements Comparator<E> {

    @Override
    public int compare(E e, E t1) {

        InteractionHistory ih1 = (InteractionHistory)e ;
        InteractionHistory ih2 = (InteractionHistory)t1 ;
        if(ih1.getErrorValue() < ih2.getErrorValue())
            return -1;
        if(ih1.getErrorValue().equals(ih2.getErrorValue()))
            return 0 ;
        return 1 ;

    }
}
