package fr.univlyon1.selfsupervised.dataConstructors;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.environment.space.SpecificObservation;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.ObservationsReplay.SpecificObservationReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.LSTM;
import fr.univlyon1.networks.LSTMMeanPooling;
import fr.univlyon1.selfsupervised.dataTransfer.DataBuilder;
import fr.univlyon1.selfsupervised.dataTransfer.DataList;
import fr.univlyon1.selfsupervised.dataTransfer.DataTarget;
import fr.univlyon1.selfsupervised.dataTransfer.ModelBasedData;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import javax.xml.bind.annotation.XmlTransient;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.NavigableSet;
import java.util.SortedSet;

@Getter
@Setter
public class LstmDataConstructor<A> extends DataConstructor<A>{
    protected SequentialExperienceReplay<A> timeEp;
    protected SpecificObservationReplay<A> labelEp;
    protected UniformIntegerDistribution srd ;
    protected LSTM lstm ;
    protected int timeDifficulty ;

    public LstmDataConstructor(LSTM lstm , SequentialExperienceReplay<A> timeEp, SpecificObservationReplay<A> labelEp, Configuration conf, ActionSpace<A> actionSpace, ObservationSpace observationSpace, SupervisedConfiguration conf2){
        super(conf,actionSpace,observationSpace,conf2);
        this.timeEp = timeEp ;
        this.labelEp = labelEp ;
        this.configuration = conf ;
        this.batchSize = conf2.getBatchSize() ;
        this.actionSpace = actionSpace ;
        this.observationSpace = observationSpace ;
        this.numberMax = conf2.getNumberMaxInputs();
        this.sequenceSize = conf.getForwardTime();
        this.dataBuilder = new DataBuilder<A>(conf2.getDataBuilder(),this);
        this.srd = new UniformIntegerDistribution(0,conf2.getTimeDifficulty()) ;
        //this.srd = new UniformIntegerDistribution(0,timeDifficulty) ;
        this.configuration2 = conf2 ;
        this.lstm = lstm ;
        this.timeDifficulty=0 ;


    }

    public ModelBasedData construct(){
        ArrayList<ArrayList<Interaction<A>>> total = new ArrayList<>();
        ArrayList<Integer> backwardsNumber = new ArrayList<>(); // nombre de backward pour chaque batch
        ArrayList<DataTarget> labelisation = new ArrayList<>();
        int forward = 0 ; // Maximum de taille de séquence temporelle
        int backward = 0 ; // Total de données labellisées, donc total de backpropagation
        int numRows = Math.min(this.timeEp.getSize(),this.batchSize);
        int size = this.observationSpace.getShape()[0]+this.actionSpace.getSize();

        //double timeDifficulty = this.timeDifficulty ;
        while(backward < numRows) {
            //int timeDifficulty = /*this.configuration2.getTimeDifficulty();*/this.srd.sample();
            /*int timeDuration = this.srd.sample();
            this.timeEp.setSequenceSize(timeDuration);
            this.timeEp.setBackpropSize(timeDuration);
            */
            this.timeEp.setBackpropSize(this.sequenceSize);
            this.timeEp.setSequenceSize(this.sequenceSize);
            if (this.timeEp.initChoose()) {
                // choix des interactions
                Interaction<A> interaction = (Interaction<A>)this.timeEp.chooseInteraction();
                ArrayList<Interaction<A>> observations = new ArrayList<>();
                while (interaction != null) {
                    observations.add(interaction);
                    interaction = this.timeEp.chooseInteraction();
                }
                int timeDifficulty = this.srd.sample();
                //DataTarget lab = this.choosePrediction(observations,this.timeDifficulty);
                DataTarget lab = this.choosePrediction(observations,timeDifficulty);
                if(lab == null)
                    continue ;
                labelisation.add(lab);
                //predictions.addAll(this.choosePrediction(observations));
                forward = Math.max(forward,this.timeEp.getForwardNumber());
                total.add(observations);
                int back= this.timeEp.getBackpropNumber();
                backwardsNumber.add(back);
                backward+=back;
            }else{
                if(total.size() == 0){
                    return null;
                }else{
                    break ;
                }
            }
        }
        int forwardInputs = forward;//-1 ;
        //contruction des INDArrays
        int totalBatchs = total.size();
        //backward = backward - totalBatchs;
        INDArray inputs = Nd4j.zeros(totalBatchs,size,forwardInputs);// On avait besoin de la taille maximale du forward
        INDArray masks = Nd4j.zeros(totalBatchs,forwardInputs);
        INDArray maskLabel  ;
        if(this.lstm instanceof LSTMMeanPooling){
            maskLabel = Nd4j.ones(totalBatchs,1);
        }else
            maskLabel = Nd4j.zeros(totalBatchs*forwardInputs,1);


        INDArray labels = Nd4j.zeros(backward,labelisation.get(0).getLabels().size(1));
        INDArray addings = Nd4j.zeros(backward,this.dataBuilder.getNumAddings());

        int cursorBackward = 0 ;
        for(int batch = 0 ; batch < total.size() ; batch++){ // Insertion des batchs
            ArrayList<Interaction<A>> observations = total.get(batch);
            int numberObservation = observations.size() ;//Nombre données temporelles
            int numBackwards = backwardsNumber.get(batch); // Nombre de backwards parmi ces données
            int start = forward - numberObservation ;
            int cursorForward = 0 ;
            for(int temporal =0 ; temporal < forward; temporal++){ // INsertion temporelles
                if(temporal < start)
                    continue ;
                Interaction<A> interact = observations.get(cursorForward);
                INDArray action = (INDArray) this.actionSpace.mapActionToNumber(interact.getAction());
                int indice = - numberObservation +numBackwards + cursorForward;
                if(indice >= 0) {
                    labels.put(new INDArrayIndex[]{NDArrayIndex.point(cursorBackward),NDArrayIndex.all()},labelisation.get(cursorBackward).getLabels());
                    addings.put(new INDArrayIndex[]{NDArrayIndex.point(cursorBackward),NDArrayIndex.all()},labelisation.get(cursorBackward).constructAddings());
                    cursorBackward++ ;
                }

                if(!(this.lstm instanceof LSTMMeanPooling) && temporal >= forwardInputs-numBackwards){//+1 ){//&& temporal < forwardInputs){
                    //System.out.println(totalBatchs*temporal + batch);
                    maskLabel.put(new INDArrayIndex[]{NDArrayIndex.point(totalBatchs*temporal + batch),NDArrayIndex.all()},Nd4j.ones(1));
                }

                //if(temporal <= forwardInputs) {
                    INDArrayIndex[] indexMask = new INDArrayIndex[]{NDArrayIndex.point(batch),NDArrayIndex.point(temporal)};
                    masks.put(indexMask, Nd4j.ones(1));

                    INDArrayIndex[] indexs = new INDArrayIndex[]{NDArrayIndex.point(batch), NDArrayIndex.all(), NDArrayIndex.point(temporal)};
                    inputs.put(indexs, Nd4j.concat(1, interact.getObservation(), action));
                //}
                cursorForward++ ;
            }
        }
        return new ModelBasedData(inputs, addings, labels, masks,maskLabel,forwardInputs,totalBatchs);
    }

    public DataTarget choosePrediction(ArrayList<Interaction<A>> observations, int dt ) {
        if(this.configuration2.getDataBuilder().equals("DataReward")){
            return this.dataBuilder.build(null,observations.get(observations.size()-1),null);
        }
        Interaction<A> last = observations.get(observations.size()-1);
        Interaction<A> chosen = this.chooseStudiedInteraction(observations,dt);
        if(chosen == null)
            return null ;
        this.labelEp.setRepere(last);
        SortedSet<SpecificObservation> set = this.labelEp.subset();
        Iterator<SpecificObservation> iterator = set.iterator();
        long id = -1 ;
        SpecificObservation spo = null ;
        int cpt = 0 ;
        while(id != chosen.getIdObserver()){ // Trouver la première notification du véhicule choisi
            if(!iterator.hasNext() || cpt > 50){
                //System.out.println("not found " + id+ " vs "+chosen.getIdObserver());
                return null ;
            }
            spo = iterator.next() ;
            id = spo.getId() ;
            cpt++ ;
        }
        //System.out.println("found "+id+ " vs "+chosen.getIdObserver());
        Double elapseTime = (spo.getOrderedNumber()-observations.get(observations.size()-1).getTime()-30.)/30.;
        //this.timeEp.setSequenceSize(this.sequenceSize);
        return this.dataBuilder.build(spo,chosen,elapseTime);
    }


    protected Interaction<A> chooseStudiedInteraction(ArrayList<Interaction<A>> observations, int dt){
        Double tmpT = 0. ;
        this.cursor = observations.size()-1;
        while(tmpT < dt && cursor >=0 && (observations.size() - cursor) < this.numberMax){ // Choix de véhicule qu'on prédit
            tmpT += observations.get(cursor).getDt() ;
            cursor -- ;
        }
        if(cursor==-1)
            return null ;
        return observations.get(cursor);
    }




}
