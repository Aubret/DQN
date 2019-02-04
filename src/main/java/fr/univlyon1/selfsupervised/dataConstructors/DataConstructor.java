package fr.univlyon1.selfsupervised.dataConstructors;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.ObservationsReplay.SpecificObservationReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.LSTM;
import fr.univlyon1.selfsupervised.dataTransfer.DataBuilder;
import fr.univlyon1.selfsupervised.dataTransfer.ModelBasedData;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public abstract class DataConstructor<A> {

    /**
     * Preprocessor in order to get data
     */
    protected Configuration configuration ;
    protected SupervisedConfiguration configuration2 ;
    protected int batchSize ;
    protected ActionSpace<A> actionSpace ;
    protected ObservationSpace observationSpace ;
    protected int numberMax ;
    protected int sequenceSize ;
    protected DataBuilder<A> dataBuilder ;
    protected int cursor ;


    public DataConstructor(Configuration conf, ActionSpace<A> actionSpace, ObservationSpace observationSpace, SupervisedConfiguration conf2){
        this.configuration = conf ;
        this.batchSize = conf2.getBatchSize() ;
        this.actionSpace = actionSpace ;
        this.observationSpace = observationSpace ;
        this.numberMax = conf2.getNumberMaxInputs();
        this.sequenceSize = conf.getForwardTime();
        this.dataBuilder = new DataBuilder<A>(conf2.getDataBuilder(),this);
        this.configuration2 = conf2 ;

        //this.srd = new UniformIntegerDistribution(0,conf2.getTimeDifficulty()) ;
    }

    public abstract ModelBasedData construct() ;

}
