package fr.univlyon1.memory.filters;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.interactions.Replayable;

import java.util.ArrayList;
import java.util.Stack;

public class IdFilter<A>  {

    /**
     * Changement d'ordre des éléments tout en enlevant les éléments les même ids de véhicule qui sont anciens.
     * On garde alors seulement le plus récent.
     * @param interactions
     * @return
     */
    public Stack<Replayable<A>> filter(ArrayList<Interaction<A>> interactions){
        ArrayList<Long> ids = new ArrayList<>();
        Stack<Replayable<A>> filteredInteractions = new Stack<>();
        for(int i = interactions.size()-1 ; i >=  0; i++){
            if(!ids.contains(interactions.get(i).getIdObserver())) {
                ids.add(interactions.get(i).getIdObserver());
                filteredInteractions.push(interactions.get(i));
            }
        }
        return filteredInteractions ;
    }
}
