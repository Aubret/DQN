package main.java.fr.univlyon1.memory;

import main.java.fr.univlyon1.environment.Interaction;

import java.util.List;

public abstract class ExperienceReplay<A> {

    protected List<Interaction<A>> memory ;
    protected int maxSize ;
    public ExperienceReplay(int maxSize){
        this.maxSize = maxSize ;
    }

    public abstract void addInteraction(Interaction<A> interaction);
    public abstract Interaction<A> chooseInteraction();
    public abstract List<Interaction<A>> getMemory();
    public abstract void resetMemory();

    public int getMaxSize() {
        return maxSize;
    }

    public void setMaxSize(int maxSize) {
        this.maxSize = maxSize;
    }
}
