package fr.univlyon1.memory.filters;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.interactions.Replayable;

import java.util.ArrayList;
import java.util.Stack;

public class NoFilter<A> implements Filter<A> {

    @Override
    public Stack<Replayable<A>> filter(ArrayList<Interaction<A>> interactions) {
        ArrayList<Long> ids = new ArrayList<>();
        Stack<Replayable<A>> filteredInteractions = new Stack<>();
        for(int i = interactions.size()-1 ; i >=  0; i--){
                filteredInteractions.push(interactions.get(i));
        }
        return filteredInteractions ;
    }
}
