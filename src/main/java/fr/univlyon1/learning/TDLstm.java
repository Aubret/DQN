package fr.univlyon1.learning;

import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.environment.Interaction;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.LSTM;
import fr.univlyon1.networks.Mlp;
import fr.univlyon1.networks.StateApproximator;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;

@Getter
@Setter
public class TDLstm<A> extends TD<A> {

    protected StateApproximator observationApproximator ;
    protected StateApproximator cloneObservationApproximator ;
    protected Approximator targetActorApproximator ;
    protected Approximator criticApproximator ;
    protected Approximator targetCriticApproximator ;
    protected Approximator cloneCriticApproximator ;
    public boolean replay = true ;

    protected Object state ;

    protected double cpt = 0 ;
    protected double cumulScoreUp=0;
    protected double cumulObservation= 0;

    protected int time = 200;
    protected int cpt_time = 0 ;
    protected boolean t = true ;

    protected Double scoreI ;
    protected SequentialExperienceReplay<A> experienceReplay;
    protected int iterations ;

    public TDLstm(double gamma, Learning<A> learning, SequentialExperienceReplay<A> experienceReplay, int iterations, Approximator criticApproximator, Approximator cloneCriticApproximator, StateApproximator observationApproximator,StateApproximator cloneObservationApproximator) {
        super(gamma, learning);
        this.iterations = iterations ;
        this.experienceReplay = experienceReplay ;
        this.observationApproximator = observationApproximator ;
        this.cloneObservationApproximator = cloneObservationApproximator ;
        //this.targetObservationApproximator = observationApproximator.clone(false);
        this.targetActorApproximator = this.learning.getApproximator().clone(false);
        this.criticApproximator = criticApproximator ;
        this.approximator = criticApproximator;
        this.targetCriticApproximator = criticApproximator.clone(false);
        this.cloneCriticApproximator = cloneCriticApproximator ; // Le clône permet de traiter deux fontions de pertes différentes.
    }

    @Override
    public INDArray behave(INDArray input){
        if(this.state != null)
            this.observationApproximator.setMemory(this.state);
        INDArray state = this.observationApproximator.getOneResult(input);
        //INDArray state= res.get(new NDArrayIndex(0), NDArrayIndex.all(), new NDArrayIndex(res.size(2)-1));
        //state = state.reshape(state.shape()[0],state.shape()[1]);
        //this.memoryAfter = this.observationApproximator.getMemory();
        INDArray action = this.learning.getApproximator().getOneResult(state);
        if(this.lastInteraction != null) {
            this.lastInteraction.setSecondState(state);
            this.lastInteraction.setSecondAction(this.learning.getActionSpace().mapNumberToAction(action));
        }
        return action ;
    }

    @Override
    public void step(INDArray observation, A action,Double time) {
        //this.lastInteraction.setSecondAction(action);
        this.lastInteraction = new Interaction<A>(action,observation);
        //this.lastInteraction.setMemoryBefore(this.memoryBefore);
        //this.lastInteraction.setMemoryAfter(this.memoryAfter);
        //this.lastInteraction.setState(this.state);
        this.lastInteraction.setTime(time);
    }

    @Override
    public void evaluate(INDArray input, Double reward) { // Store transistions
        if(this.lastInteraction != null) { // Avoir des interactions complètes
            this.lastInteraction.setSecondObservation(input);
            this.lastInteraction.setReward(reward);
            this.experienceReplay.addInteraction(this.lastInteraction);
        }
    }

    /**
     * Extract mini batch
     */
    @Override
    public void learn(){
        this.state = this.observationApproximator.getMemory();
        if(this.replay)
            this.learn_replay();
        //else
        //    learn_online();
    }

    protected void learn_replay(){
        for(int i = 0;i < this.iterations ; i++){
            if(this.experienceReplay.initChoose()) {
                // choix des interactions
                Interaction<A> lastInteraction = null ;
                Interaction<A> interaction = this.experienceReplay.chooseInteraction();
                ArrayList<Interaction<A>> observations = new ArrayList<>() ;
                observations.add(interaction);
                while(interaction != null ){
                    observations.add(interaction);
                    lastInteraction = interaction ;
                    interaction = this.experienceReplay.chooseInteraction();
                }
                int size = observations.size() ;
                int[] dimensions = new int[]{1,lastInteraction.getObservation().shape()[1],size};
                this.observationApproximator.setBackpropNumber(this.experienceReplay.getBackpropNumber());
                this.observationApproximator.setForwardNumber(this.experienceReplay.getForwardNumber());
                INDArray inputs = Nd4j.zeros(dimensions);
                for(int j = 0 ; j < observations.size(); j++){
                    INDArrayIndex[] indexs = new INDArrayIndex[]{ NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.point(j)};
                    inputs.put(indexs, observations.get(j).getObservation());
                }
                // Apprentissage : besoin de l'état
                INDArray state_label = this.observationApproximator.error(inputs, null, 1);
                //state_label= state_label.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(state_label.size(2)-1));

                //Apprentissage critique
                INDArray labels = this.labelize(lastInteraction);
                INDArray action = (INDArray)this.learning.getActionSpace().mapActionToNumber(lastInteraction.getAction());
                INDArray inputCritics = Nd4j.concat(1,state_label,action);
                INDArray epsilon = this.learn_critic(inputCritics,labels,1);
                //Apprentissage politique
                int sizeObservation =state_label.size(1);
                int sizeAction = this.learning.getActionSpace().getSize() ;
                this.learn_actor(state_label,sizeObservation,sizeAction,1); // Important entre la propagation de l'observation et la backpropagation du gradient
                //Apprentissage des observations
                INDArray epsilonObservation = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0,sizeObservation));
                this.learn_observator(inputs,epsilonObservation,1,action,labels);
                this.cpt_time++;
            }
        }
    }

    protected void learn_observator(INDArray inputs, INDArray labels, int numRows,INDArray action,INDArray label){
        this.observationApproximator.learn(inputs,labels,1);
        if(this.cpt_time%this.time == 0){
            INDArray firstval = ((Mlp) this.criticApproximator).getValues().detach();
            INDArray s1 = firstval.sub(labels);
            Double val1 = s1.muli(s1).meanNumber().doubleValue();

            LSTM test = (LSTM)this.observationApproximator;
            INDArray res = test.getOneTrainingResult(inputs);
            INDArray inputAction = Nd4j.concat(1,res,action);
            INDArray newVal = this.criticApproximator.getOneResult(inputAction);
            INDArray s2 = newVal.sub(labels);
            Double val2 =s2.muli(s2).meanNumber().doubleValue();
            double meanScore = val1-val2 ;
            cumulObservation+=meanScore ;
            System.out.println(meanScore + " -- " +cumulObservation);

        }

    }

    protected INDArray learn_critic(INDArray inputs, INDArray labels, int numRows){
        //System.out.println("----");
        //INDArray epsilonObsAct = (INDArray)this.criticApproximator.learn(inputs, labels, numRows); // Critic learning
        INDArray epsilon = (INDArray)this.criticApproximator.learn(inputs, labels, numRows);// Critic learning
        INDArray scores = this.criticApproximator.getScoreArray();
        this.experienceReplay.setError(scores);

        if(this.cpt_time%this.time == 0){
            System.out.println("-------------");
            INDArray firstval = ((Mlp) this.criticApproximator).getValues().detach();
            INDArray s1 = firstval.sub(labels);
            Double val1 = s1.muli(s1).meanNumber().doubleValue();
            INDArray newVal = this.criticApproximator.getOneResult(inputs);
            INDArray s2 = newVal.sub(labels);
            Double val2 =s2.muli(s2).meanNumber().doubleValue();
            double meanScore = val1-val2 ;
            cumulScoreUp+=meanScore ;
            System.out.println(meanScore + " -- " +cumulScoreUp);

        }
        return epsilon;
    }

    protected INDArray learn_actor(INDArray observations, int sizeObservation, int numColumns, int numRows){
        INDArray action = this.learning.getApproximator().getOneResult(observations); // L'action du policy network
        INDArray inputAction = Nd4j.concat(1, observations, action);
        INDArray epsilonObsAct = this.cloneCriticApproximator.error(inputAction, Nd4j.create(new double[]{0}), numRows); // erreur
        INDArray epsilonAction = epsilonObsAct.get(NDArrayIndex.all(), NDArrayIndex.interval(sizeObservation, sizeObservation+numColumns));
        INDArray old =null;
        if(this.cpt_time%this.time == 0) {
            old = this.cloneCriticApproximator.getOneResult(inputAction);
        }
        this.learning.getApproximator().learn(observations, epsilonAction, numRows); //Policy learning

        if(this.cpt_time%this.time == 0 ) {
            action = this.learning.getApproximator().getOneResult(observations); // L'action du policy network
            inputAction = Nd4j.concat(1, observations, action);
            INDArray intermediaire = this.cloneCriticApproximator.getOneResult(inputAction).subi(old);//must be positive
            Number mean = intermediaire.meanNumber();
            cpt += mean.doubleValue();
            System.out.println(mean + " -- " + cpt);
        }
        return epsilonObsAct ;
    }


    protected INDArray labelize(Interaction<A> interaction){
        // Les états précédents sont dans la mémoire de l'approximateur
        this.cloneObservationApproximator.setParams(this.observationApproximator.getParams());
        this.cloneObservationApproximator.setMemory(this.observationApproximator.getMemory());
        INDArray stateLabel = this.cloneObservationApproximator.getOneResult(interaction.getSecondObservation());
        //stateLabel = stateLabel.reshape(stateLabel.shape()[0],stateLabel.shape()[1]);

        INDArray action = this.targetActorApproximator.getOneResult(stateLabel);
        INDArray entryCriticTarget = Nd4j.concat(1,stateLabel, action) ;
        INDArray res = this.targetCriticApproximator.getOneResult(entryCriticTarget);
        res = res.muli(this.gamma).addi(interaction.getReward()) ;
        return res ;
    }




    public void epoch() {
        /*double alpha = 0.001;
        targetActorApproximator.getParams().muli(1. - alpha).addi(this.learning.getApproximator().getParams().mul(alpha));
        targetCriticApproximator.getParams().muli(1. - alpha).addi(this.criticApproximator.getParams().mul(alpha));
        this.targetObservationApproximator.getParams().muli(1. - alpha).addi(this.observationApproximator.getParams().mul(alpha));
        */
        targetActorApproximator.setParams(this.learning.getApproximator().getParams());
        targetCriticApproximator.setParams(this.criticApproximator.getParams());
        //targetObservationApproximator.setParams(this.observationApproximator.getParams());
    }





    /*protected void learn_online(){
        if(this.lastInteraction == null)
            return ;
        INDArray labels = this.labelize(this.lastInteraction);
        int numRows = 1;
        INDArray inputAction = Nd4j.concat(1,this.lastInteraction.getState(), (INDArray)this.learning.getActionSpace().mapActionToNumber(this.lastInteraction.getAction())); ;
        INDArray epsilon = this.learn_critic(inputAction, labels,numRows);
        INDArray epsilonActor = this.learn_actor(this.lastInteraction.getState(),1);

        int sizeObservation = this.observationApproximator.numOutput() ;
        INDArray epsilonObservation = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0,sizeObservation));
        this.observationApproximator.setMemory(this.memoryBefore);
        this.observationApproximator.learn(this.lastInteraction.getObservation(),epsilonObservation,numRows);
        this.cpt_time++ ;
    }*/




    /*protected INDArray learn_actor(INDArray states,int numRows){
        int sizeObservation = states.size(1);

        INDArray action = this.learning.getApproximator().getOneResult(states); // L'action du policy network
        INDArray inputAction = Nd4j.concat(1, states, action);
        INDArray epsilonObsAct = this.cloneCriticApproximator.error(inputAction, Nd4j.create(new double[]{0}), numRows); // erreur
        INDArray epsilonAction = epsilonObsAct.get(NDArrayIndex.all(), NDArrayIndex.interval(sizeObservation, epsilonObsAct.size(1)));
        INDArray old =null;
        if(this.cpt_time%this.time == 0) {
            old = this.cloneCriticApproximator.getOneResult(inputAction);
        }

        this.learning.getApproximator().learn(states, epsilonAction, numRows); //Policy learning

        if(this.cpt_time%this.time == 0 ) {
            action = this.learning.getApproximator().getOneResult(states); // L'action du policy network
            inputAction = Nd4j.concat(1, states, action);
            INDArray intermediaire = this.cloneCriticApproximator.getOneResult(inputAction).subi(old);//must be positive
            Number mean = intermediaire.meanNumber();
            cpt += mean.doubleValue();
            System.out.println(mean + " -- " + cpt);
        }
        return epsilonObsAct ;
    }

    protected INDArray learn_critic(INDArray inputs, INDArray labels, int numRows){
        //System.out.println("----");
        //INDArray epsilonObsAct = (INDArray)this.criticApproximator.learn(inputs, labels, numRows); // Critic learning
        INDArray epsilon = (INDArray)this.criticApproximator.learn(inputs, labels, numRows);// Critic learning
        //INDArray scores = this.criticApproximator.getScoreArray();
        //this.experienceReplay.setError(scores);

        if(this.cpt_time%this.time == 0){
            System.out.println("-------------");
            INDArray firstval = ((Mlp) this.criticApproximator).getValues();
            INDArray s1 = firstval.sub(labels);
            Double val1 = s1.muli(s1).meanNumber().doubleValue();
            INDArray newVal = this.criticApproximator.getOneResult(inputs);
            INDArray s2 = newVal.sub(labels);
            Double val2 =s2.muli(s2).meanNumber().doubleValue();
            double meanScore = val1-val2 ;
            cumulScoreUp+=meanScore ;
            System.out.println(meanScore + " -- " +cumulScoreUp);
        }
        return epsilon ;
    }


    protected INDArray labelize(Interaction<A> interaction){
        //this.targetObservationApproximator.setMemory(this.memoryAfter);
        this.observationApproximator.setMemory(this.memoryAfter);
        //INDArray state = this.targetObservationApproximator.getOneResult(interaction.getSecondObservation());
        INDArray state = this.observationApproximator.getOneResult(interaction.getSecondObservation());
        INDArray actionTarget = this.targetActorApproximator.getOneResult(state);
        //INDArray actionTarget = Nd4j.zeros(this.learning.getActionSpace().getSize()).add(0.1);
        INDArray entryCriticTarget = Nd4j.concat(1,state, actionTarget) ;
        INDArray res = this.targetCriticApproximator.getOneResult(entryCriticTarget);
        res = res.muli(this.gamma).addi(interaction.getReward()) ;
        return res ;
    }*/





}
