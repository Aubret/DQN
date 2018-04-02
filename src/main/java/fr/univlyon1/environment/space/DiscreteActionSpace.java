package fr.univlyon1.environment.space;

import java.util.Random;

public class DiscreteActionSpace<A> extends ActionSpace<A> {

    private Random random ;

    public DiscreteActionSpace(long seed){
        super();
        this.random = new Random(seed);
    }

    public void addAction(A action){
        this.actions.add(action);
    }

    public A mapNumberToAction(Object number){
        return this.actions.get((int)number);
    }

    @Override
    public Object mapActionToNumber(A action) {
        for(int i = 0 ; i < this.actions.size(); i++){
            if(this.actions.get(i) == action)
                return i ;
        }
        return -1 ;
    }

    @Override
    public Object randomAction() {
        return random.nextInt(this.getSize());
        //return null;
    }
}
