with import <nixpkgs> {};
with pkgs.python36Packages;

let
   gym = buildPythonPackage rec {
     pname = "gym";
     version = "0.9.4";
     name = "${pname}-${version}";

     src = pkgs.fetchurl {
       url = "https://pypi.python.org/packages/f8/9f/b50f4c04a97e316ebfccae3104e5edbfe7bc1c687ee9ebeca6fa6343d197/gym-0.9.4.tar.gz";
       sha256 = "121qq4al4in5pmq4am8aa2g70476yp2kvk2bb0y29cdsj2kirycl";
     };

     doCheck = false;
     buildInputs = [
		    python36Full
                    cudatoolkit
                    cudnn
                    python36Packages.requests
                    python36Packages.pyglet
                    python36Packages.six
                    python36Packages.numpy
                    python36Packages.scipy
                   ];
   };

in

buildPythonPackage{
    name = "deeprl";
    buildInputs = [ 
                    python36Full
                    python36Packages.matplotlib
                    python36Packages.setuptools
                    python36Packages.numpy
                    python36Packages.cloudpickle
                    gym
                   ]; 
  shellHook = ''
  # set SOURCE_DATE_EPOCH so that we can use python wheels
  SOURCE_DATE_EPOCH=$(date +%s)
  CPATH=$CPATH:~/.local/include
  LIBRARY_PATH=$LIBRARY_PATH:~/.local/lib
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib

  '';

}

