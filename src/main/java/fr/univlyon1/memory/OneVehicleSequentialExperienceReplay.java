package fr.univlyon1.memory;

import fr.univlyon1.environment.interactions.Interaction;

import java.util.ArrayList;
import java.util.Stack;

public class OneVehicleSequentialExperienceReplay<A> extends SequentialExperienceReplay<A> {

    protected Stack<Interaction<A>> filteredInteractions ;

    public OneVehicleSequentialExperienceReplay(int maxSize, ArrayList<String> file, int sequenceSize, int backpropSize, long seed, Integer forwardSize) {
        super(maxSize, file, sequenceSize, backpropSize, seed, forwardSize);
    }

    public boolean initChoose() { // Toujours appeler avant les chooseInteraction
        boolean init = super.initChoose();
        if (!init)
            return false;
        ArrayList<Interaction<A>> interactions = new ArrayList<>();
        Interaction<A> interaction = super.chooseInteraction();
        interactions = new ArrayList<>();
        while (interaction != null) {
            interactions.add(interaction);
            interaction = super.chooseInteraction();
        }
        this.filter(interactions);
        return true;
    }

    @Override
    public Interaction<A> chooseInteraction() {
        return this.filteredInteractions.pop();
    }

    protected void filter(ArrayList<Interaction<A>> interactions){
        ArrayList<Long> ids = new ArrayList<>();
        this.filteredInteractions = new Stack<>();
        for(int i = this.interactions.size()-1 ; i >=  0; i++){
            if(!ids.contains(this.interactions.get(i).getIdObserver())) {
                this.filteredInteractions.push(this.interactions.get(i));
            }
        }
    }

}
