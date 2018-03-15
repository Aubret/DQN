package main.java.fr.univlyon1.agents;

import main.java.fr.univlyon1.environment.Interaction;
import main.java.fr.univlyon1.environment.Observation;

public interface AgentRL<A> {
    Object control(Double reward,Observation observation);
    void stop();
}
