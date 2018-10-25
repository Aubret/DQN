package fr.univlyon1.memory;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.interactions.Replayable;
import fr.univlyon1.memory.filters.Filter;
import fr.univlyon1.memory.filters.IdFilter;
import fr.univlyon1.memory.filters.NoFilter;
import play.mvc.WebSocket;

import java.util.ArrayList;
import java.util.Stack;

public class OneVehicleSequentialExperienceReplay<A> extends SequentialExperienceReplay<A> {

    protected Stack<Replayable<A>> filteredInteractions ;
    protected Filter<A> filter ;


    public OneVehicleSequentialExperienceReplay(int maxSize, ArrayList<String> file, int sequenceSize, int backpropSize, long seed, Integer forwardSize) {
        super(maxSize, file, sequenceSize, backpropSize, seed, forwardSize);
        this.filter = new NoFilter<A>();
    }

    public boolean initChoose(){ // Toujours appeler avant les chooseInteraction
        if(!super.initChoose())
            return false ;
        ArrayList<Interaction<A>> interactions = constructInteractions() ;
        this.filteredInteractions = this.filter.filter(interactions);
        this.forwardNumber = this.filteredInteractions.size();
        return true ;
    }

    protected ArrayList<Interaction<A>> constructInteractions(){
        this.tmp = new ArrayList<>();
        ArrayList<Interaction<A>> interactions = new ArrayList<>();
        Interaction<A> interaction = super.chooseInteraction();
        while (interaction != null) {
            interactions.add(interaction);
            interaction = super.chooseInteraction();
        }
        return interactions;
    }

    @Override
    public Interaction<A> chooseInteraction() {
        return this.filteredInteractions.isEmpty() ? null : (Interaction<A>)this.filteredInteractions.pop();
    }


    @Override
    public Stack<Replayable<A>> lastInteraction(){
        double time = 0. ;
        if(this.interactions.size() < 2){
            System.out.println("error");
            return null ;
        }

        ArrayList<Interaction<A>> lasts = new ArrayList<>();
        int cursor = this.interactions.size()-1;
        while(time < this.sequenceSize){
            time+=this.interactions.get(cursor).getDt();
            cursor -- ;
        }
        for(int i = cursor; i < this.interactions.size();i++){
            lasts.add(this.interactions.get(i));
        }
        return this.filter.filter(lasts);
    }
}
