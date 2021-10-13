from MissingHandler import MissingHandler

df = pd.read_excel(r'path_to_excel_file')

X = v1.iloc[:, 0].copy()
y = v1.iloc[:, [-1]].copy()

# average strategy
handler = MissingHandler(X, y, method='average')
X, y = handler.transform()

# median strategy
handler = MissingHandler(X, y, method='median')
X, y = handler.transform()

# mode strategy
handler = MissingHandler(X, y, method='mode')
X, y = handler.transform()

# geometric mean strategy
handler = MissingHandler(X, y, method='geomean')
X, y = handler.transform()

# delete strategy
handler = MissingHandler(X, y, method='delete')
X, y = handler.transform()

# min strategy
handler = MissingHandler(X, y, method='min')
X, y = handler.transform()

# max strategy
handler = MissingHandler(X, y, method='max')
X, y = handler.transform()

# last_observe strategy
handler = MissingHandler(X, y, method='last_observe')
X, y = handler.transform()

# next_observe strategy
handler = MissingHandler(X, y, method='next_observe')
X, y = handler.transform()

# linear model
handler = MissingHandler(X, y, method='linear_strategy')
X, y = handler.transform()
