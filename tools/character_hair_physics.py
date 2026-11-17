"""Conservative bone hair dynamics for the two imported Random Play actors."""
import json
import bpy
import wiggle_2
from mathutils import Matrix, Vector


def configure_hair(scene, slug):
    rig, mesh = scene.sam_ellen.rig, scene.xr_ellen.face_mesh
    scene.wiggle_enable = False
    for bone in rig.data.bones:
        if hasattr(bone, 'select'):
            bone.select = False
        elif hasattr(bone, 'select_set'):
            bone.select_set(False)
    collection_name = slug + ' Hair Body Colliders'
    old = bpy.data.collections.get(collection_name)
    if old:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)
    colliders = bpy.data.collections.new(collection_name)
    scene.collection.children.link(colliders)
    allowed = {'頭', '首', '上半身', '上半身2', '左肩', '右肩',
               '左腕', '右腕', '左ひじ', '右ひじ', '左手首', '右手首'}
    for obj in list(scene.objects):
        if getattr(obj, 'mmd_type', '') != 'RIGID_BODY' or obj.mmd_rigid.type != '0':
            continue
        bone = rig.pose.bones.get(obj.mmd_rigid.bone)
        if not bone or bone.mmd_bone.name_j not in allowed:
            continue
        # Hidden metadata has stale world matrices. Reconstruct model space
        # from the local hierarchy, then bind against the bone REST matrix.
        model_matrix = obj.matrix_basis.copy()
        parent = obj.parent
        while parent and parent != rig.parent:
            model_matrix = parent.matrix_basis @ model_matrix
            parent = parent.parent
        proxy = bpy.data.objects.new(slug + ' Hair Collider ' + obj.name, obj.data.copy())
        colliders.objects.link(proxy)
        proxy.parent = rig
        proxy.parent_type = 'BONE'
        proxy.parent_bone = bone.name
        proxy.matrix_parent_inverse = Matrix.Identity(4)
        rest_parent = bone.bone.matrix_local @ Matrix.Translation((0, bone.length, 0))
        proxy.matrix_basis = rest_parent.inverted() @ model_matrix
        proxy.display_type = 'WIRE'
        proxy.hide_render = True
        proxy.hide_set(True)
        proxy['HairPhysicsCollider'] = True

    labels = {'后髪', '前发'} if slug == 'Derechina' else {'发' if slug == 'Remielle' else '髮'}
    material_ids = {i for i, m in enumerate(mesh.data.materials)
                    if m and m.get('original_material', m.name.rsplit('|', 1)[-1].strip()) in labels}
    vertices = {v for p in mesh.data.polygons if p.material_index in material_ids for v in p.vertices}
    weighted = set()
    for vi in vertices:
        for group in mesh.data.vertices[vi].groups:
            if group.weight > .005:
                weighted.add(mesh.vertex_groups[group.group].name)
    candidates = {b.name for b in rig.pose.bones if b.name in weighted and
                  b.name.startswith('Ctr_') and ('Hair' in b.name or 'Bangs' in b.name or 'SuiFa' in b.name)
                  and not b.name.endswith(('+', '錘', '先')) and 'HairTie' not in b.name}
    if slug == 'Derechina':
        candidates = {b.name for b in rig.pose.bones if b.name in weighted and
                      any(token in b.mmd_bone.name_j for token in ('前发', '髪', '穗'))
                      and not b.name.startswith(('_dummy_', '_shadow_'))}
    selected = {n for n in candidates if sum(c.name in candidates for c in rig.pose.bones[n].children) <= 2}
    profile = []
    for name in sorted(selected):
        bone = rig.pose.bones[name]
        depth, parent = 0, bone.parent
        while parent and parent.name in selected:
            depth += 1
            parent = parent.parent
        front = ('Ctr_F_' in name or 'Ctr_FL_' in name or 'Ctr_FR_' in name or
                 ('Bangs' in name and bone.bone.head_local.y < .012))
        if slug == 'Derechina':
            front = '前发' in bone.mmd_bone.name_j
        limit_angle = (.045 if front else .075) / (1 + depth * .6)
        near_body = slug == 'Remielle' and name.startswith(('Ctr_L_HairA', 'Ctr_L_HairB',
                    'Ctr_R_HairA', 'Ctr_R_HairB', 'Ctr_B_HairB'))
        if near_body:
            limit_angle = .006 / (1 + depth * .6)
        if slug == 'Derechina':
            limit_angle = (.015 if front else .025) / (1 + depth * .6)
        bone.wiggle_enable = True
        bone.wiggle_tail = True
        bone.wiggle_head = False
        bone.wiggle_chain = True
        bone.wiggle_stiff = 900 if near_body else (620 if front else 480)
        bone.wiggle_damp = 14 if near_body else (10 if front else 8)
        bone.wiggle_gravity = .025 if front else .045
        bone.wiggle_stretch = 0
        bone.wiggle_radius = .003
        bone.wiggle_mass = 1
        bone.wiggle_collider_type = 'Collection'
        bone.wiggle_collider_collection = colliders
        bone.wiggle_friction = .8
        bone.wiggle_bounce = 0
        for con in list(bone.constraints):
            if con.name.startswith('Hair safety sway'):
                bone.constraints.remove(con)
        con = bone.constraints.new('LIMIT_ROTATION')
        con.name = 'Hair safety sway - fixed length'
        con.owner_space = 'LOCAL'
        direction = bone.bone.tail_local - bone.bone.head_local
        local = bone.bone.matrix_local.to_3x3().inverted() @ direction.cross(Vector((1, 0, 0)))
        axis = max(range(3), key=lambda i: abs(local[i]))
        for i, component in enumerate('xyz'):
            angle = limit_angle if not front or i == axis else 0
            setattr(con, 'use_limit_' + component, True)
            setattr(con, 'min_' + component, -angle)
            setattr(con, 'max_' + component, angle)
        profile.append(dict(bone=name, front=front, depth=depth, limit=limit_angle))
    assert selected and len(colliders.objects) >= 10
    rig.wiggle_enable = True
    rig.wiggle_mute = False
    rig.wiggle_freeze = False
    scene.wiggle.iterations = 6
    scene.wiggle_enable = True
    scene.xr_ellen.live_physics = True
    scene.render.use_lock_interface = True
    scene['CharacterHairProfile'] = json.dumps(profile)
    scene['SecondaryMotionNote'] = 'Fixed-length Wiggle hair; limited bangs and branch sway; bone-following head, neck, torso, shoulder and arm collision proxies.'
    wiggle_2.build_list()
    bpy.context.view_layer.update()
    bpy.ops.wiggle.reset()
    return dict(bones=len(selected), colliders=len(colliders.objects), profile=profile)
