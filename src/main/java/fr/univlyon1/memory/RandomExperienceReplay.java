package fr.univlyon1.memory;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.interactions.Replayable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

public class RandomExperienceReplay<A> extends ExperienceReplay<A> {
    protected ArrayList<Interaction<A>> memory ;
    private Random random ;
    public RandomExperienceReplay(int maxSize,long seed,ArrayList<String> file){
        super(maxSize,file);
        this.random = new Random(seed);
        this.resetMemory();
    }

    @Override
    public void addInteraction(Replayable<A> replayable) {
        Interaction<A> interaction = (Interaction<A>)replayable ;
        if(memory.size() == this.maxSize)
            memory.remove(0);
        //System.out.println("add"+interaction.getDt());
        memory.add(interaction);
    }



    @Override
    public Interaction<A> chooseInteraction() {
        Interaction<A> i = (Interaction<A>)this.memory.get(random.nextInt(this.memory.size()));
        //System.out.println("choose"+ i.getDt());
        return i;
    }

    public Collection<Interaction<A>> getMemory() {
        return this.memory ;
    }

    @Override
    public void resetMemory() {
        this.memory = new ArrayList<Interaction<A>>();
    }

    @Override
    public int getSize() {
        return this.memory.size();
    }

    @Override
    public void setError(INDArray errors) {
    }

    @Override
    public Stack<Replayable<A>> lastInteraction() {
        Stack<Replayable<A>> last = new Stack<>();
        last.push(this.memory.get(this.memory.size()-1));
        return last ;
    }
}
