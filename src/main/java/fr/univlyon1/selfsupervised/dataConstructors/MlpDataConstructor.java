package fr.univlyon1.selfsupervised.dataConstructors;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.environment.space.SpecificObservation;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.ObservationsReplay.SpecificObservationReplay;
import fr.univlyon1.memory.RandomExperienceReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.LSTM;
import fr.univlyon1.selfsupervised.dataTransfer.DataBuilder;
import fr.univlyon1.selfsupervised.dataTransfer.DataTarget;
import fr.univlyon1.selfsupervised.dataTransfer.ModelBasedData;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.SortedSet;

public class MlpDataConstructor<A> extends DataConstructor<A>{
    protected ExperienceReplay<A> ep ;
    protected SpecificObservationReplay<A> epObs ;

    public MlpDataConstructor(ExperienceReplay<A> ep, SpecificObservationReplay<A> epObs, Configuration conf, ActionSpace<A> actionSpace, ObservationSpace observationSpace, SupervisedConfiguration conf2) {
        super(conf, actionSpace, observationSpace, conf2);
        this.ep = ep ;
        this.epObs = epObs ;
    }

    public ModelBasedData construct() {
        int numRows = this.configuration2.getBatchSize() ;


        INDArray observations = Nd4j.zeros(numRows, observationSpace.getShape()[0]+actionSpace.getSize());
        INDArray addings= Nd4j.zeros(numRows, this.dataBuilder.getNumAddings());
        INDArray labels = Nd4j.zeros(numRows, this.dataBuilder.getNumPredicts());
        for (int i = 0; i < numRows; i++) {
            Interaction<A> interaction = (Interaction<A>) this.ep.chooseInteraction();
            DataTarget data = this.choosePrediction(interaction) ;
            while(data ==null){
                interaction = (Interaction<A>) this.ep.chooseInteraction();
                data = this.choosePrediction(interaction) ;
            }
            INDArray inputAction = Nd4j.concat(1, interaction.getObservation(), (INDArray)this.actionSpace.mapActionToNumber(interaction.getAction()));
            observations.putRow(i, inputAction);
            addings.putRow(i, data.constructAddings());
            labels.putRow(i, data.getLabels());
        }
        return new ModelBasedData(observations,addings,labels,null, null,1,numRows) ;
    }

    public DataTarget choosePrediction(Interaction<A> observation ) {
        if(this.configuration2.getDataBuilder().equals("DataReward")){
            return this.dataBuilder.build(null,observation,null);
        }
        this.epObs.setRepere(observation);
        SortedSet<SpecificObservation> set = this.epObs.subset();
        Iterator<SpecificObservation> iterator = set.iterator();
        long id = -1 ;
        SpecificObservation spo = null ;
        int cpt = 0 ;
        //System.out.println("--------");
        //System.out.println(observation.getTime());
        while(id != observation.getIdObserver()){ // Trouver la première notification du véhicule choisi
            if(!iterator.hasNext() || cpt > 50){
                //System.out.println("not found " + id+ " vs "+observation.getIdObserver());
                return null ;
            }
            spo = iterator.next() ;
            //System.out.println(spo.getOrderedNumber());
            id = spo.getId() ;
            cpt++ ;
        }
        Double elapseTime = (spo.getOrderedNumber()-observation.getTime()-30.)/30.;
        return this.dataBuilder.build(spo,observation,elapseTime);
    }
}


