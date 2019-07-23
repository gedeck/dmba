'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and 
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck 
'''
import io
import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz
try:
  from IPython.display import Image
except ImportError:
  Image = None
try:
  import pydotplus
except ImportError:
  pydotplus = None


def liftChart(predicted, title='Decile Lift Chart', labelBars=True, ax=None, figsize=None):
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
    meanResponse = meanPercentile / predicted.mean()
    meanResponse.index = (meanResponse.index + 1) * 10

    ax = meanResponse.plot.bar(color='C0', ax=ax, figsize=figsize)
    ax.set_ylim(0, 1.12 * meanResponse.max() if labelBars else None)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Lift')
    if title:
        ax.set_title(title)

    if labelBars:
        for p in ax.patches:
            ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x(), p.get_height() + 0.1))
    return ax


def gainsChart(gains, color=None, label=None, ax=None, figsize=None):
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
    
    # filled polygon shows the perfect model
    # ax.fill([0, nActual, nTotal], [0, nActual, nActual], color='#eeeeee')

    ax = gains_df.plot(x='records', y='cumGains', color=color, label=label, legend=False, 
                       ax=ax, figsize=figsize)
    # Add line for perfect model
    # ax.plot([0, nActual, nTotal], [0, nActual, nActual], linestyle='--', color='r')

    # Add line for random gain
    ax.plot([0, nTotal], [0, nActual], linestyle='--', color='k')
    ax.set_xlabel('# records')
    ax.set_ylabel('# cumulative gains')
    return ax


def plotDecisionTree(decisionTree, feature_names=None, class_names=None, impurity=False, label='root',
                     max_depth=None, rotate=False, pdfFile=None):
    """ Create a plot of the scikit-learn decision tree and show in the Jupyter notebooke 
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
    if pydotplus is None:
        return 'You need to install pydotplus to visualize decision trees'
    if Image is None:
        return 'You need to install ipython to visualize decision trees'
    if class_names is not None:
        class_names = [str(s) for s in class_names]  # convert to strings
    dot_data = io.StringIO()
    export_graphviz(decisionTree, feature_names=feature_names, class_names=class_names, impurity=impurity,
                    label=label, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    max_depth=max_depth, rotate=rotate)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    if pdfFile is not None:
        graph.write_pdf(str(pdfFile))
    return Image(graph.create_png())

# Taken from scikit-learn documentation
def textDecisionTree(decisionTree, indent='  ', as_ratio=True):
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
        if (children_left[node_id] != children_right[node_id]):
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
              value = [[round(vi/sum(v), 3) for vi in v] for v in value]
            rep.append(f'{common} leaf node: {value}')
        else:
            rule = f'{children_left[i]} if {feature[i]} <= {threshold[i]} else to node {children_right[i]}'
            rep.append(f'{common} test node: go to node {rule}')
    return '\n'.join(rep)
