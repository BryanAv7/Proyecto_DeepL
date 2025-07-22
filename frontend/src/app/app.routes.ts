import { Routes } from '@angular/router';
import { ValidacionComponent } from './validacion/validacion.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { LoginComponent } from './login/login.component';
import { ChatComponent } from './chat/chat.component';
import { AuthGuard } from '../auth.guard'; 
import { LoginGuard } from '../login.guard'; // importa el guard creado

export const routes: Routes = [
  { path: '', redirectTo: 'login', pathMatch: 'full' },  
  { path: 'login', component: LoginComponent, canActivate: [LoginGuard] },  // protejo login para no entrar si está logueado
  { path: 'dashboard', component: DashboardComponent, canActivate: [AuthGuard] },
  { path: 'validacion', component: ValidacionComponent },
  { path: 'chat', component: ChatComponent, canActivate: [AuthGuard] },      // protejo chat también
  { path: '**', redirectTo: '' },  
];
