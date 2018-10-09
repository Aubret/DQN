package fr.univlyon1.selfsupervised.modelLearning;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.networks.LSTM2D;
import fr.univlyon1.networks.Mlp;

public abstract class Learner<A> {
    protected SupervisedConfiguration conf;
    protected ActionSpace<A> actionSpace ;
    protected ObservationSpace observationSpace ;
    protected Configuration configuration ;
    protected long seed ;

    public Learner(SupervisedConfiguration supervisedConfiguration, Configuration configuration, ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed ){
        this.conf = supervisedConfiguration ;
        this.actionSpace = actionSpace ;
        this.seed=seed ;
        this.observationSpace = observationSpace ;
        this.configuration = configuration ;

    }


    public abstract void learn();
    public abstract void stop();

}
