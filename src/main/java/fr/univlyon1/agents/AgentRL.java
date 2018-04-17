package fr.univlyon1.agents;

import fr.univlyon1.environment.space.Observation;

public interface AgentRL<A> {
    A control(Double reward,Observation observation);
    void stop();
}
