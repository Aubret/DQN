package fr.univlyon1.agents;

import fr.univlyon1.environment.Observation;

public interface AgentRL<A> {
    Object control(Double reward,Observation observation);
    void stop();
}
