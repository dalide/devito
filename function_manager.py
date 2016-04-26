import cgen_wrapper as cgen


class FunctionManager(object):
    # Order of names in the following list is important.
    # The resulting code blocks would be placed in the same
    # order as they appear here

    libraries = ['cassert', 'cstdlib', 'cmath', 'iostream',
                 'fstream', 'vector', 'cstdio', 'string', 'inttypes.h']

    def __init__(self, function_descriptors):
        self.function_descriptors = function_descriptors
        self._defines = []

    def includes(self):
        statements = []
        statements += self._defines
        statements += [cgen.Include(s) for s in self.libraries]
        return cgen.Module(statements)

    def add_define(self, name, text):
        self._defines.append(cgen.Define(name, text))

    def generate(self):
        statements = [self.includes()]
        statements += [self.process_function(m) for m in self.function_descriptors]
        return cgen.Module(statements)

    def process_function(self, function_descriptor):
        return cgen.FunctionBody(self.generate_function_signature(function_descriptor),
                                 self.generate_function_body(function_descriptor))

    def generate_function_signature(self, function_descriptor):
        function_params = []
        for param in function_descriptor.params:
            try:
                # Assume the parameter is a matrix parameter
                param_vec_def = cgen.Pointer(cgen.POD(param['dtype'], param['name']+"_vec"))
                function_params = function_params + [param_vec_def]
            except TypeError:  # Unless it is a value parameter, in which case make sure
                assert(len(param) == 2)

        function_params += [cgen.POD(type_label, name) for type_label, name in function_descriptor.value_params]
        return cgen.Extern("C",
                           cgen.FunctionDeclaration(cgen.Value('int', function_descriptor.name),
                                                    function_params)
                           )

    def generate_function_body(self, function_descriptor):
        statements = [cgen.POD(var[0], var[1]) for var in function_descriptor.local_vars]
        for param in function_descriptor.params:
            try:
                num_dim = len(param['shape'])
                arr = "".join(
                    ["[%d]" % (param['shape'][i])
                     for i in range(1, num_dim)]
                )
                cast_pointer = cgen.Initializer(
                    cgen.Value("double", "(*%s)%s" % (param['name'], arr)),
                    '(%s (*)%s) %s' % ("double", arr, param['name']+"_vec")
                )
                statements.append(cast_pointer)
            except TypeError:  # It might be a value parameter
                pass  # Do nothing, in that case
        statements.append(function_descriptor.body)
        statements.append(cgen.Statement("return 0"))
        return cgen.Block(statements)