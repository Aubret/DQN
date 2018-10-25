package fr.univlyon1.memory.filters;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.interactions.Replayable;

import java.util.ArrayList;
import java.util.Stack;

public interface Filter<A> {
    Stack<Replayable<A>> filter(ArrayList<Interaction<A>> interactions);
}
