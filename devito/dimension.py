import sympy
from cached_property import cached_property

from devito.arguments import DimensionArgProvider
from devito.types import Symbol

__all__ = ['Dimension', 'SpaceDimension', 'TimeDimension', 'SteppingDimension']


class Dimension(sympy.Symbol, DimensionArgProvider):

    is_Space = False
    is_Time = False

    is_Stepping = False
    is_Lowered = False

    """Index object that represents a problem dimension and thus
    defines a potential iteration space.

    :param name: Name of the dimension symbol.
    :param reverse: Traverse dimension in reverse order (default False)
    :param spacing: Optional, symbol for the spacing along this dimension.
    """

    def __new__(cls, name, **kwargs):
        newobj = sympy.Symbol.__new__(cls, name)
        newobj.reverse = kwargs.get('reverse', False)
        newobj.spacing = kwargs.get('spacing', sympy.Symbol('h_%s' % name))
        return newobj

    def __str__(self):
        return self.name

    @cached_property
    def symbolic_size(self):
        """The symbolic size of this dimension."""
        return Symbol(name=self.size_name)

    @cached_property
    def symbolic_start(self):
        """
        The symbol defining the iteration start for this dimension.
        """
        return Symbol(name=self.start_name)

    @cached_property
    def symbolic_end(self):
        """
        The symbol defining the iteration end for this dimension.
        """
        return Symbol(name=self.end_name)

    @property
    def symbolic_extent(self):
        """Return the extent of the loop over this dimension.
        Would be the same as size if using default values """
        _, start, end = self.rtargs
        return (self.symbolic_end - self.symbolic_start)

    @property
    def limits(self):
        _, start, end = self.rtargs
        return (self.symbolic_start, self.symbolic_end, 1)

    @property
    def size_name(self):
        return "%s_size" % self.name

    @property
    def start_name(self):
        return "%s_s" % self.name

    @property
    def end_name(self):
        return "%s_e" % self.name

    def _hashable_content(self):
        return super(Dimension, self)._hashable_content() +\
            (self.reverse, self.spacing)

    def argument_defaults(self, size=None):
        """
        Returns a map of default argument values defined by this symbol.

        :param size: Optional, known size as provided by data-carrying symbols
        """
        return {self.start_name: 0, self.end_name: size, self.size_name: size}

    def argument_values(self, **kwargs):
        """
        Returns a map of argument values after evaluating user input.

        :param kwargs: Dictionary of user-provided argument overrides.
        """
        values = {}

        if self.start_name in kwargs:
            values[self.start_name] = kwargs.pop(self.start_name)

        if self.end_name in kwargs:
            values[self.end_name] = kwargs.pop(self.end_name)

        if self.name in kwargs:
            values[self.end_name] = kwargs.pop(self.name)

        return values

class SpaceDimension(Dimension):

    is_Space = True

    """
    Dimension symbol to represent a space dimension that defines the
    extent of physical grid. :class:`SpaceDimensions` create dedicated
    shortcut notations for spatial derivatives on :class:`Function`
    symbols.

    :param name: Name of the dimension symbol.
    :param reverse: Traverse dimension in reverse order (default False)
    :param spacing: Optional, symbol for the spacing along this dimension.
    """


class TimeDimension(Dimension):

    is_Time = True

    """
    Dimension symbol to represent a dimension that defines the extent
    of time. As time might be used in different contexts, all derived
    time dimensions should inherit from :class:`TimeDimension`.

    :param name: Name of the dimension symbol.
    :param reverse: Traverse dimension in reverse order (default False)
    :param spacing: Optional, symbol for the spacing along this dimension.
    """


class SteppingDimension(Dimension):

    is_Stepping = True

    """
    Dimension symbol that defines the stepping direction of an
    :class:`Operator` and implies modulo buffered iteration. This is most
    commonly use to represent a timestepping dimension.

    :param parent: Parent dimension over which to loop in modulo fashion.
    """

    def __new__(cls, name, parent, **kwargs):
        newobj = sympy.Symbol.__new__(cls, name)
        assert isinstance(parent, Dimension)
        newobj.parent = parent
        newobj.modulo = kwargs.get('modulo', 2)

        # Inherit time/space identifiers
        cls.is_Time = parent.is_Time
        cls.is_Space = parent.is_Space

        return newobj

    @property
    def reverse(self):
        return self.parent.reverse

    @property
    def spacing(self):
        return self.parent.spacing

    def _hashable_content(self):
        return (self.parent._hashable_content(), self.modulo)

    @property
    def symbolic_start(self):
        """
        The symbol defining the iteration start for this dimension.

        note ::

        Internally we always define symbolic iteration ranges in terms
        of the parent variable.
        """
        return self.parent.symbolic_start

    @property
    def symbolic_end(self):
        """
        The symbol defining the iteration end for this dimension.

        note ::

        Internally we always define symbolic iteration ranges in terms
        of the parent variable.
        """
        return self.parent.symbolic_end

    def argument_defaults(self, size=None):
        """
        Returns a map of default argument values defined by this symbol.

        :param size: Optional, known size as provided by data-carrying symbols

        note ::

        A :class:`SteppingDimension` neither knows it's size nor it's
        iteration end point. So all we can provide is a starting point.
        """
        return {self.parent.start_name: 0}

    def argument_values(self, **kwargs):
        """
        Returns a map of argument values after evaluating user input.

        :param kwargs: Dictionary of user-provided argument overrides.
        """
        values = self.parent.argument_values(**kwargs)

        if self.start_name in kwargs:
            values[self.parent.start_name] = kwargs.pop(self.start_name)

        if self.end_name in kwargs:
            values[self.parent.end_name] = kwargs.pop(self.end_name)

        return values

class LoweredDimension(Dimension):

    is_Lowered = True

    """
    Dimension symbol representing a modulo iteration created when
    resolving a :class:`SteppingDimension`.

    :param stepping: :class:`SteppingDimension` from which this
                     :class:`Dimension` originated.
    :param offset: Offset value used in the modulo iteration.
    """

    def __new__(cls, name, stepping, offset, **kwargs):
        newobj = sympy.Symbol.__new__(cls, name)
        assert isinstance(stepping, SteppingDimension)
        newobj.stepping = stepping
        newobj.offset = offset
        return newobj

    @property
    def origin(self):
        return self.stepping + self.offset

    @property
    def size(self):
        return self.stepping.size

    @property
    def reverse(self):
        return self.stepping.reverse

    def _hashable_content(self):
        return sympy.Symbol._hashable_content(self) + (self.stepping, self.offset)
