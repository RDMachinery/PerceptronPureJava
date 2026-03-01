package org.anticml;

import org.anticml.event.LinearRegressionEvent;

/**
 *
 * @author Mario
 */
public interface LinearRegressionListener {

    public void handleEvent(LinearRegressionEvent e);
}
