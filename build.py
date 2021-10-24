#   -*- coding: utf-8 -*-
from pybuilder.core import init, use_plugin

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.flake8")
use_plugin("python.coverage")
use_plugin("python.distutils")


name = "Labor_Scheduler"
default_task = "publish"


@init
def set_properties(project):
    project.depends_on_requirements("requirements.txt")
    project.set_property("coverage_threshold_warn", 85)
    project.set_property("coverage_break_build", False)
    project.set_property("dir_source_unittest_python", "src/main/python/models")
    pass
