package fr.univlyon1.memory.sumTree;

import fr.univlyon1.memory.prioritizedExperienceReplay.InteractionHistory;

public interface INode<A> {
    void insert(InteractionHistory<A> ih);
}
