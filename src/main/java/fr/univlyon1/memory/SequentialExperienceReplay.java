package fr.univlyon1.memory;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.interactions.Replayable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class SequentialExperienceReplay<A> extends ExperienceReplay<A>{
    protected ArrayList<Interaction<A>> interactions ;
    protected ArrayList<Interaction<A>> tmp ;
    protected INDArray constructedData ;
    protected Integer cursor=0;
    protected int sequenceSize ;
    protected int backpropSize ;
    protected Random random ;

    protected int forwardNumber ;
    protected int backpropNumber;
    protected double startTime ;

    protected int minForward ;

    public SequentialExperienceReplay(int maxSize, ArrayList<String> file,int sequenceSize, int backpropSize,long seed,Integer forwardSize){
        super(maxSize,file);
        this.resetMemory();
        this.sequenceSize = sequenceSize ;
        this.backpropSize = backpropSize ;
        this.random = new Random(seed);
        this.minForward = forwardSize ;
    }

    @Override
    public void addInteraction(Replayable<A> replayable) {
        Interaction<A> interaction = (Interaction)replayable ;
        if(this.interactions.size() == this.maxSize)
            this.interactions.remove(this.interactions.get(0));
        this.interactions.add(interaction);
    }
    public void addInteraction(Interaction<A> interaction,double error) {
        if(this.interactions.size() == this.maxSize)
            this.interactions.remove(this.interactions.get(0));
        this.interactions.add(interaction);
    }

    public boolean initChoose(){ // Toujours appeler avant les chooseInteraction
        if(this.interactions.size() <= minForward)
            return false ;
        if(this.interactions.get(this.interactions.size()-1).getTime() - this.interactions.get(0).getTime() < this.sequenceSize )
            return false ;
        // On vérifie qu ele curseur actuel suffit à proposer une séquence complète
        Interaction<A> start = this.choose();
        Double dt = this.interactions.get(this.interactions.size()-1).getTime() - start.getTime() ;
        int cpt = 0 ;
        while(dt < this.sequenceSize || this.interactions.size() - cursor <= minForward || this.interactions.get(cursor+1).getTime()-start.getTime() > this.sequenceSize){
            if(cpt == 10){
                cursor=0 ; // On veut limiter le nombre de recherches aléatoires
                break;
            }else {
                start = this.choose();
                dt = this.interactions.get(this.interactions.size()-1).getTime() - start.getTime() ;
                //cursor = 0; // On a déjà vérifié que c'était possible avec 0
                cpt++ ;
            }
        }
        this.startTime = start.getTime() ;
        this.backpropNumber = 0 ;
        this.forwardNumber = 0 ;
        this.tmp = new ArrayList<>();
        return true ;
    }

    protected Interaction<A> choose(){
        cursor = this.random.nextInt(this.interactions.size()-1);
        return this.interactions.get(cursor);
    }

    @Override
    public Interaction<A> chooseInteraction() {
        Interaction<A> choose = this.interactions.get(this.cursor);
        Double dt = choose.getTime() - startTime;
        if(dt > sequenceSize)
            return null;
        this.tmp.add(choose);
        this.forwardNumber++ ;
        this.cursor++ ;
        return choose;
    }

    @Override
    public void resetMemory() {
        this.interactions = new ArrayList<>();
        this.tmp = new ArrayList<>();
    }

    @Override
    public int getSize() {
        return this.interactions.size();
    }

    @Override
    public AbstractCollection<? extends Replayable<A>> getMemory() {
        return this.interactions ;
    }

    @Override
    public void setError(INDArray errors) {

    }

    public void setError(INDArray errors, ArrayList<Integer> backpropNumber, int backward,ArrayList<ArrayList<Interaction<A>>> total) {

    }


    public int getForwardNumber(){
        return this.forwardNumber ;
    }

    public int getBackpropNumber(){
        Double endTime = this.tmp.get(this.tmp.size()-1).getTime() ;
        for(int i = this.tmp.size()-1 ; i >= 0 ; i-- ) {
            if (endTime - this.tmp.get(i).getTime() > this.backpropSize) {
                return Math.max(this.minForward,Math.min(this.backpropNumber,this.minForward));
                //return this.backpropNumber;
            } else
                this.backpropNumber++;
        }
        return Math.max(this.minForward,Math.min(this.backpropNumber,this.minForward));
        //return this.backpropNumber;
    }

    public int getSequenceSize() {
        return sequenceSize;
    }

    public void setSequenceSize(int sequenceSize) {
        this.sequenceSize = sequenceSize;
    }

    public int getBackpropSize() {
        return backpropSize;
    }

    public void setBackpropSize(int backpropSize) {
        this.backpropSize = backpropSize;
    }

    public INDArray getConstructedData() {
        return constructedData;
    }

    public void setConstructedData(INDArray constructedData) {
        this.constructedData = constructedData;
    }
}
