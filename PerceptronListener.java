package org.anticml;

import org.anticml.event.PerceptronEvent;

public abstract class PerceptronListener {

	public void trainingStarted(PerceptronEvent e) { }
	public void trainingComplete(PerceptronEvent e) { }
	public void trainingInProgress(PerceptronEvent e) { }
}
