# ShapeAssembly: Learning to Generate Programs for 3D Shape Structure Synthesis 

By [R. Kenny Jones](https://rkjones4.github.io/), [Theresa Barton](https://github.com/tbarton16), [Xianghao Xu](https://www.linkedin.com/in/xianghao-xu-8b1024a6/), [Kai Wang](https://kwang-ether.github.io/),
[Ellen Jiang](https://ellenjiang.com/), [Paul Guerrero](https://paulguerrero.net/), [Niloy J. Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/), and [Daniel Ritchie](https://dritchie.github.io/)

![Overview](https://rkjones4.github.io/img/shapeAssembly/teaser.png)

We present a deep generative model which learns to write novel programs in ShapeAssembly, a domain-specific language for modeling 3D shape structures. Executing a ShapeAssembly program produces a shape composed of a hierarchical connected assembly of part proxies cuboids. Our method develops a well-formed latent space that supports interpolations between programs. Above, we show one such interpolation, and also visualize the geometry these programs produce when executed. In the last column, we manually edit the continuous parameters of a generated program, in order to produce a variant geometric structure with new topology.

## About the paper

Paper: https://rkjones4.github.io/pdf/shapeAssembly.pdf

Accepted by [Siggraph Asia 2020](https://sa2020.siggraph.org/).

Project Page: https://rkjones4.github.io/shapeAssembly.html

## Citations
```
   @article{jones2020shapeAssembly,
	title={ShapeAssembly: Learning to Generate Programs for 3D Shape Structure Synthesis},
	author={Jones, R. Kenny and Barton, Theresa and Xu, Xianghao and Wang, Kai and Jiang, Ellen and Guerrero, Paul and Mitra, Niloy and Ritchie, Daniel},
	journal={ACM Transactions on Graphics (TOG), Siggraph Asia 2020},
	volume={39},
	number={6},
	pages={Article 234},
 	year={2020},
	publisher={ACM}
    }
```

## ShapeAssembly DSL

The ShapeAssembly DSL is defined in code/ShapeAssembly.py . Within this file, the ShapeAssembly class is the entrypoint for interacting with the language.

For instance, the file code/data/chair/173.txt describes a chair from our training set. To execute this program, and write the resulting shape to executed.obj, from the command line we can run:
```
python3 ShapeAssembly.py run data/chair/173.txt executed.obj
```
from the code directory. Or from a python shell we can run:
```
> from ShapeAssembly import ShapeAssembly
> sa = ShapeAssembly()
> lines = sa.load_lines('data/chair/173.txt')
> sa.run(lines, 'executed.obj')
```

The interace for executing a non-hierarchical program is very similiar (just replace run with run_local). The file code/data/examples/ex-0.txt contains a non-hierarchical ShapeAssembly program describing a table-top. 

## Model Training/Evaluation

The model for generating ShapeAssembly programs is defined in code/model_prog.py

The command to train our model on the chair category can be run from the code directory as:
```
python3 model_prog.py --dataset_path data/chair/ --category chair --exp_name chair_exp
```
Training results from this run will be placed in model_output/chair_exp.

We also include pre-trained models for the chair, table and storage categories. These can be found in the code/model_output/pre_* folders.

To generate shapes from these pre-trained models the following command structure can be used:
```
python3 model_prog.py --mode eval_gen --num_gen 10 --exp_name pre_chair --model_name pre_chair --load_epoch 119
```
This will generate 10 chair-programs and output them in code/model_output/gen_pre_chair. 

Other Relevant modeling files:

parse_prog.py -> functions for tensorizing/un-tensorizing text-based ShapeAssembly programs

losses.py -> define losses used during training

sem_valid.py -> logic implementing semantic validity checks used when decoding ShapeAssembly programs at eval time

metrics.py -> functions for computing metrics of the generative model's performance. pointnet_classification.py, valid.py and voxelize.py are metric helper files.

## Data 

We provide our parsed ShapeAssembly datasets for chairs, tables and storage categories in the code/data folder.

To re-generate these datasets, or to run our program parsing procedure on other part-graphs, please use the gen_data.py script in the code directory.

If [part-graph json files](https://github.com/daerduoCarey/structurenet/tree/master/data/partnetdata) are placed in /home/{USER}/pnhier/{category} for a given category, ShapeAssembly programs can be generated with a command like:

```
python3 gen_data.py {category}
```

Additional program checks can be enforced by running code/clean_data.py on the resulting parses. 

Other relevant program parsing files:

json_parse.py -> parse a hierarchical part-graph into collection of attributes needed for its corresponding ShapeAssembly program. Point cloud intersection logic takes place in intersect.py. Symmetry detection logic takes place in symmetry.py.

generate.py -> takes a collection of attributes and formulate a ShapeAssembly program. Attachment statement ordering logic is within attach_order.py. Detection of squeeze attach operators takes place in squeeze.py.

## Dependencies

Code was tested on Ubuntu 18.04 with pytorch 3.7. env.yml should contain a further list of all the conda packages/versions used to run the code.

