package fr.univlyon1.memory.filters;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.interactions.Replayable;

import java.util.ArrayList;
import java.util.Stack;

public class IdFilter<A> implements Filter<A>{

    /**
     * Changement d'ordre des éléments tout en enlevant les éléments les même ids de véhicule qui sont anciens.
     * On garde alors seulement le plus récent.
     * @param interactions
     * @return
     */
    public Stack<Replayable<A>> filter(ArrayList<Interaction<A>> interactions){
        ArrayList<Long> ids = new ArrayList<>();
        //StringBuffer str = new StringBuffer("");
        /*StringBuffer str2 = new StringBuffer("");

        for(Interaction<A> ite : interactions){
            str.append(" ");
            str.append(ite.getIdObserver());
        }*/


        Stack<Replayable<A>> filteredInteractions = new Stack<>();
        for(int i = interactions.size()-1 ; i >=  0; i--){
            if(!ids.contains(interactions.get(i).getIdObserver())) {
                ids.add(interactions.get(i).getIdObserver());
                /*str.append(" ");
                str.append(interactions.get(i).getTime());*/
                filteredInteractions.push(interactions.get(i));
            }
        }

        /*for(Replayable<A> ite : filteredInteractions){
            str2.append(" ");
            str2.append(((Interaction)ite).getTime());
            //str2.append(((Interaction)ite).getIdObserver());
        }
        /*System.out.println("-----");
        System.out.println(str.toString());*/
        //System.out.println(str.toString());
        return filteredInteractions ;
    }
}
