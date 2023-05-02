'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019-2023 Galit Shmueli, Peter C. Bruce, Peter Gedeck
'''
import io
import os
from tempfile import TemporaryDirectory
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.tree import export_graphviz

hasGraphviz = False
try:
    import graphviz
    hasGraphviz = True
except ImportError:
    pass

try:
    from IPython.display import Image
    hasImage = True
except ImportError:
    hasImage = False

def liftChart(predicted: pd.Series, *, title: str = 'Decile Lift Chart', labelBars: bool = True,
              ax: Any = None, figsize: Any = None) -> Any:
    """ Create a lift chart using predicted values

    Input:
        predictions: must be sorted by probability
        ax (optional): axis for matplotlib graph
        title (optional): set to None to suppress title
        labelBars (optional): set to False to avoid mean response labels on bar chart
    """
    # group the sorted predictions into 10 roughly equal groups and calculate the mean
    groups = [int(10 * i / len(predicted)) for i in range(len(predicted))]
    meanPercentile = predicted.groupby(groups).mean()
    # divide by the mean prediction to get the mean response
    meanResponse = meanPercentile / predicted.mean()  # type: ignore
    meanResponse.index = (meanResponse.index + 1) * 10

    ax = meanResponse.plot.bar(color='C0', ax=ax, figsize=figsize)
    ax.set_ylim(0, 1.12 * meanResponse.max() if labelBars else None)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Lift')
    if title:
        ax.set_title(title)

    if labelBars:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}', (p.get_x(), p.get_height() + 0.1))
    return ax


def gainsChart(gains: pd.Series, color: str = 'C0', label: Optional[str] = None,
               ax: Any = None, figsize: Any = None) -> Any:
    """ Create a gains chart using predicted values

    Input:
        gains: must be sorted by probability
        color (optional): color of graph
        ax (optional): axis for matplotlib graph
        figsize (optional): size of matplotlib graph
    """
    nTotal = len(gains)  # number of records
    nActual = gains.sum()  # number of desired records

    # get cumulative sum of gains and convert to percentage
    cumGains = pd.concat([pd.Series([0]), gains.cumsum()])  # Note the additional 0 at the front
    gains_df = pd.DataFrame({'records': list(range(len(gains) + 1)), 'cumGains': cumGains})

    ax = gains_df.plot(x='records', y='cumGains', color=color, label=label, legend=False,
                       ax=ax, figsize=figsize)

    # Add line for random gain
    ax.plot([0, nTotal], [0, nActual], linestyle='--', color='k')
    ax.set_xlabel('# records')
    ax.set_ylabel('# cumulative gains')
    return ax


def plotDecisionTree(decisionTree: Any, *, feature_names: Optional[List[str]] = None,
                     class_names: Optional[List[str]] = None, impurity: bool = False,
                     label: str = 'root', max_depth: Optional[int] = None, rotate: bool = False,
                     pdfFile: Optional[os.PathLike] = None) -> Any:
    """ Create a plot of the scikit-learn decision tree and show in the Jupyter notebook

    Input:
        decisionTree: scikit-learn decision tree
        feature_names (optional): variable names
        class_names (optional): class names, only relevant for classification trees
        impurity (optional): show node impurity
        label (optional): only show labels at the root
        max_depth (optional): limit
        rotate (optional): rotate the layout of the graph
        pdfFile (optional): provide pathname to create a PDF file of the graph
    """
    if not hasGraphviz:
        return 'You need to install graphviz to visualize decision trees'
    if not hasImage and not pdfFile:
        return 'You need to install Image and/or graphviz to visualize decision trees'
    if class_names is not None:
        class_names = [str(s) for s in class_names]  # convert to strings
    dot_data = io.StringIO()
    export_graphviz(decisionTree, feature_names=feature_names, class_names=class_names, impurity=impurity,
                    label=label, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    max_depth=max_depth, rotate=rotate)
    graph = graphviz.Source(dot_data.getvalue())
    with TemporaryDirectory() as tempdir:
        if pdfFile is not None:
            graph.render('dot', directory=tempdir, format='pdf', outfile=pdfFile)
        if hasImage:
            return Image(graph.render('dot', directory=tempdir, format='png'))
        return None

# Taken from scikit-learn documentation


def textDecisionTree(decisionTree: Any, indent: str = '  ', as_ratio: bool = True) -> str:  # noqa: FBT
    """ Create a text representation of the scikit-learn decision tree

    Input:
        decisionTree: scikit-learn decision tree
        as_ratio: show the composition of the leaf nodes as ratio (default) instead of counts
        indent: indentation (default two spaces)
    """
    n_nodes = decisionTree.tree_.node_count
    children_left = decisionTree.tree_.children_left
    children_right = decisionTree.tree_.children_right
    feature = decisionTree.tree_.feature
    threshold = decisionTree.tree_.threshold
    node_value = decisionTree.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    rep = []
    for i in range(n_nodes):
        common = f'{node_depth[i] * indent}node={i}'
        if is_leaves[i]:
            value = node_value[i]
            if as_ratio:
                value = [[round(vi / sum(v), 3) for vi in v] for v in value]
            rep.append(f'{common} leaf node: {value}')
        else:
            rule = f'{children_left[i]} if {feature[i]} <= {threshold[i]} else to node {children_right[i]}'
            rep.append(f'{common} test node: go to node {rule}')
    return '\n'.join(rep)
