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
import fr.univlyon1.selfsupervised.dataTransfer.ModelBasedData;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.sql.Array;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.NavigableSet;
import java.util.SortedSet;

public class LstmDataConstructors<A> {
    private SequentialExperienceReplay<A> timeEp;
    private SpecificObservationReplay<A> labelEp;
    private Configuration configuration ;
    private int batchSize ;
    private ActionSpace<A> actionSpace ;
    private ObservationSpace observationSpace ;
    private int timeDifficulty ;
    private int numberMax ;
    private int sequenceSize ;


    public LstmDataConstructors(SequentialExperienceReplay<A> timeEp, SpecificObservationReplay<A> labelEp, Configuration conf, ActionSpace<A> actionSpace, ObservationSpace observationSpace, SupervisedConfiguration conf2){
        this.timeEp = timeEp ;
        this.labelEp = labelEp ;
        this.configuration = conf ;
        this.batchSize = conf2.getBatchSize() ;
        this.actionSpace = actionSpace ;
        this.observationSpace = observationSpace ;
        this.timeDifficulty = conf2.getTimeDifficulty();
        this.numberMax = conf2.getNumberMaxInputs();
        this.sequenceSize = conf.getForwardTime();

    }

    public ModelBasedData construct(){
        ArrayList<ArrayList<Interaction<A>>> total = new ArrayList<>();
        ArrayList<Integer> backwardsNumber = new ArrayList<>(); // nombre de backward pour chaque batch
        ArrayList<DataList> labelisation = new ArrayList<>();
        int forward = 0 ; // Maximum de taille de séquence temporelle
        int backward = 0 ; // Total de données labellisées, donc total de backpropagation
        int numRows = Math.min(this.timeEp.getSize(),this.batchSize);
        int size = this.observationSpace.getShape()[0]+this.actionSpace.getSize();

        this.timeEp.setSequenceSize(this.sequenceSize + this.timeDifficulty);
        while(backward < numRows) {
            if (this.timeEp.initChoose()) {
                // choix des interactions
                Interaction<A> interaction = (Interaction<A>)this.timeEp.chooseInteraction();
                ArrayList<Interaction<A>> observations = new ArrayList<>();
                while (interaction != null) {
                    observations.add(interaction);
                    interaction = (Interaction<A>)this.timeEp.chooseInteraction();
                }
                DataList lab = this.choosePrediction(observations,this.timeDifficulty);
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
        int forwardInputs = forward-1 ;
        //contruction des INDArrays
        int totalBatchs = total.size();
        backward = backward - totalBatchs;
        INDArray inputs = Nd4j.zeros(totalBatchs,size,forwardInputs);// On avait besoin de la taille maximale du forward
        INDArray masks = Nd4j.zeros(totalBatchs,forwardInputs);
        INDArray maskLabel = Nd4j.zeros(totalBatchs*forwardInputs,1);
        INDArray labels = Nd4j.zeros(backward,labelisation.get(0).getObservation().getLabels().size(1));
        INDArray secondInputs = Nd4j.zeros(backward,this.numAddings());

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
                if(indice > 0) {
                    labels.put(new INDArrayIndex[]{NDArrayIndex.point(cursorBackward),NDArrayIndex.all()},labelisation.get(cursorBackward).getObservation().getLabels());
                    secondInputs.put(new INDArrayIndex[]{NDArrayIndex.point(cursorBackward),NDArrayIndex.all()},labelisation.get(cursorBackward).constructAddings());
                    cursorBackward++ ;
                }

                if(temporal >= forwardInputs-numBackwards+1 && temporal < forwardInputs){
                    maskLabel.put(new INDArrayIndex[]{NDArrayIndex.point(totalBatchs*temporal + batch),NDArrayIndex.all()},Nd4j.ones(1));
                }

                if(temporal < forwardInputs) {
                    INDArrayIndex[] indexMask = new INDArrayIndex[]{NDArrayIndex.point(batch),NDArrayIndex.point(temporal)};
                    masks.put(indexMask, Nd4j.ones(1));

                    INDArrayIndex[] indexs = new INDArrayIndex[]{NDArrayIndex.point(batch), NDArrayIndex.all(), NDArrayIndex.point(temporal)};
                    inputs.put(indexs, Nd4j.concat(1, interact.getObservation(), action));
                }
                cursorForward++ ;
            }
        }
        return new ModelBasedData(inputs, secondInputs, labels, masks,maskLabel);
    }

    public DataList choosePrediction(ArrayList<Interaction<A>> observations,int dt ) {
        Double tmpT = 0. ;
        int cursor = observations.size();
        Interaction<A> last = observations.get(cursor-1);
        while(tmpT < dt && cursor >=0 && (observations.size() - cursor) < this.numberMax){ // Choix de véhicule qu'on prédit
            cursor -- ;
            tmpT += observations.get(cursor).getDt() ;
        }
        Interaction<A> chosen = observations.get(cursor);
        this.labelEp.setRepere(last);
        SortedSet<SpecificObservation> set = this.labelEp.subset();
        Iterator<SpecificObservation> iterator = set.iterator();
        long id = -1 ;
        SpecificObservation spo = null ;
        while(id != chosen.getIdObserver()){ // Trouver la première notification du véhicule choisi
            if(!iterator.hasNext()){
                return null ;
            }
            spo = iterator.next() ;
            id = spo.getId() ;
        }

        Double numMaxNorm = Integer.valueOf(this.numberMax).doubleValue()/2. ;
        Double normalizedId = (Integer.valueOf(cursor).doubleValue() - numMaxNorm)/numMaxNorm ;
        Double elapseTime = (spo.getOrderedNumber()-observations.get(observations.size()-1).getTime()-30.)/30.;
        this.timeEp.setSequenceSize(this.sequenceSize);
        return new DataList(spo,chosen,elapseTime,normalizedId);
    }


    public int numAddings(){
        return 2;
    }
    @Getter
    @Setter
    public class DataList {
        public SpecificObservation observation ;
        public Interaction<A> predictions ;
        public Double extratime ;
        public Double normalizedId ;

        public DataList(SpecificObservation observation, Interaction<A> predictions, Double extratime, Double normalizedId){
            this.observation = observation ;
            this.predictions = predictions ;
            this.extratime = extratime ;
            this.normalizedId = normalizedId ;
        }

        public INDArray constructAddings(){
            return Nd4j.create(new double[]{this.extratime, this.normalizedId});
        }

    }
}
