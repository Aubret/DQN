package fr.univlyon1.memory;

import fr.univlyon1.environment.Interaction;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class RandomExperienceReplay<A> extends ExperienceReplay<A> {
    protected List<Interaction<A>> memory ;
    private Random random ;
    public RandomExperienceReplay(int maxSize,long seed){
        super(maxSize);
        this.random = new Random(seed);
        this.resetMemory();
    }

    @Override
    public void addInteraction(Interaction<A> interaction) {
        LinkedList<Interaction<A>> memory = (LinkedList<Interaction<A>>)this.memory ;
        if(memory.size() == this.maxSize)
            memory.removeLast();
        memory.addFirst(interaction);
    }



    @Override
    public Interaction<A> chooseInteraction() {
        Interaction<A> i = this.memory.get(random.nextInt(this.memory.size()));
        return i;
    }

    public List<Interaction<A>> getMemory() {
        return this.memory ;
    }

    @Override
    public void resetMemory() {
        this.memory = new LinkedList<Interaction<A>>();
    }

    @Override
    public int getSize() {
        return this.memory.size();
    }
}
