package main.java.fr.univlyon1.environment;

import java.util.Random;

public class ContinuousActionSpace<A> extends ActionSpace<A> {
    private Random random;

    public ContinuousActionSpace() {
        super();
        random = new Random();
    }

    public void addAction(A action) {
        this.actions.add(action);
    }

    public A mapNumberToAction(Object number) {
        return this.actions.get((int)number);
    }

    public A randomAction() {
        return this.actions.get(random.nextInt(this.getSize()));
    }


    @Override
    public int mapActionToNumber(A action) {
        for(int i = 0 ; i < this.actions.size(); i++){
            if(this.actions.get(i) == action)
                return i ;
        }
        return -1 ;
    }

}
