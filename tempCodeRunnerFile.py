    model_graphviz = model.to_graphviz()

    # Plot the model.
    model_graphviz.draw("sachs.png", prog="dot")

    # Other file formats can also be specified.
    model_graphviz.draw("sachs.pdf", prog="dot")
    model_graphviz.draw("sachs.svg", prog="dot")